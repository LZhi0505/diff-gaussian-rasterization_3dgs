int CudaRasterizer::Rasterizer::forward(
        std::function<char* (size_t)> geometryBuffer,   // 三个都是调整内存缓冲区的函数指针
        std::function<char* (size_t)> binningBuffer,
        std::function<char* (size_t)> imageBuffer,
        const int P,    // 所有高斯的个数
        int D,      // 当前的球谐阶数
        int M,      // 每个高斯的球谐系数个数=16
        const float* background,    // 背景颜色，默认为[1,1,1]，黑色
        const int width, int height,    // 图像宽、高
        const float* means3D,   // 所有高斯 中心的世界坐标
        const float* shs,       // 所有高斯的 球谐系数
        const float* colors_precomp,    // 因预计算的颜色默认是空tensor，则其传入的是一个 NULL指针
        const float* opacities, // 所有高斯的 不透明度
        const float* scales,    // 所有高斯的 缩放因子
        const float scale_modifier, // 缩放因子的调整系数
        const float* rotations,     // 所有高斯的 旋转四元数
        const float* cov3D_precomp, // 因预计算的3D协方差矩阵默认是空tensor，则其传入的是一个 NULL指针
        const float* viewmatrix,    // 观测变换矩阵，W2C
        const float* projmatrix,    // 观测变换矩阵 * 投影变换矩阵，W2NDC = W2C * C2NDC
        const float* cam_pos,       // 当前相机中心的世界坐标
        const float tan_fovx, float tan_fovy,
        const bool prefiltered,     // 预滤除的标志，默认为False
        float* out_color,       // 输出的 颜色图像，(3,H,W)
        int* radii,             // 输出的 在图像平面上的投影半径(N,)
        bool debug)     // 默认为False
{
    // 1. 计算焦距，W = 2fx * tan(Fovx/2) ==> fx = W / (2 * tan(Fovx/2))
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    // 2. 分配、初始化几何信息geomState（包含）
    size_t chunk_size = required<GeometryState>(P);     // 根据高斯的数量 P，计算存储所有高斯各参数 所需的空间大小
    char* chunkptr = geometryBuffer(chunk_size);        // 分配指定大小 chunk_size的缓冲区，即给所有高斯的各参数分配存储空间，返回指向该存储空间的指针 chunkptr
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);    // 在给定的内存块中初始化 GeometryState 结构体, 为不同成员分配空间，并返回一个初始化的实例

    if (radii == nullptr) {
        // 如果传入的、要输出的 高斯在图像平面的投影半径为 nullptr，则将其设为
        radii = geomState.internal_radii;
    }

    // 3. 定义一个三维CUDA网格 tile_grid，确定了在水平和垂直方向上需要多少个线程块来覆盖整个渲染区域，即W/16，H/16
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    // 定义每个线程块 block，确定了在水平和垂直方向上的线程数。每个线程处理一个像素，则每个线程块处理16*16个像素
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // 4. 在训练期间动态调整与图像相关的辅助缓冲区
    size_t img_chunk_size = required<ImageState>(width * height);   // 计算存储所有2D pixel的各个参数所需要的空间大小
    char* img_chunkptr = imageBuffer(img_chunk_size);                   // 给所有2D pixel的各个参数分配存储空间, 并返回存储空间的指针
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);  // 在给定的内存块中初始化 ImageState 结构体, 为不同成员分配空间，并返回一个初始化的实例

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
    }

    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    //! 1. 预处理和投影：将每个高斯投影到图像平面上、计算投影所占的tile块坐标和个数、根据球谐系数计算RGB值。 具体实现在 forward.cu/preprocessCUDA
    CHECK_CUDA(FORWARD::preprocess(
            P, D, M,
            means3D,
            (glm::vec3*)scales,
            scale_modifier,
            (glm::vec4*)rotations,
            opacities,
            shs,
            geomState.clamped,      // geomState中记录高斯是否被裁剪的标志，即某位置为 True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
            cov3D_precomp,          // 因预计算的3D协方差矩阵默认是空tensor，则传入的是一个 NULL指针
            colors_precomp,         // 因预计算的颜色默认是空tensor，则传入的是一个 NULL指针
            viewmatrix, projmatrix,
            (glm::vec3*)cam_pos,
            width, height,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            radii,              // 输出的 所有高斯 投影在当前相机图像平面的最大半径 数组
            geomState.means2D,  // 输出的 所有高斯 中心在当前相机图像平面的二维坐标 数组
            geomState.depths,   // 输出的 所有高斯 中心在当前相机坐标系下的z值 数组
            geomState.cov3D,    // 输出的 所有高斯 在世界坐标系下的3D协方差矩阵 数组
            geomState.rgb,      // 输出的 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
            geomState.conic_opacity,    // 输出的 所有高斯 2D协方差的逆 和 不透明度 数组
            tile_grid,                  // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
            geomState.tiles_touched,    // 输出的 所有高斯 在图像平面覆盖的线程块 tile的个数 数组
            prefiltered                 // 预滤除的标志，默认为False
    ), debug)

//! 2. 高斯排序和合成顺序：根据高斯距离摄像机的远近来计算每个高斯在Alpha合成中的顺序
// ---开始--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表
// Compute prefix sum over full list of touched tile counts by Gaussians
// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

// Retrieve total number of Gaussian instances to launch and resize aux buffers
// 存储所有的2D gaussian总共覆盖了多少个tile
int num_rendered;
// 将 geomState.point_offsets 数组中最后一个元素的值复制到主机内存中的变量 num_rendered
CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

size_t binning_chunk_size = required<BinningState>(num_rendered);
char* binning_chunkptr = binningBuffer(binning_chunk_size);
BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

// For each instance to be rendered, produce adequate [ tile | depth ] key
// and corresponding dublicated Gaussian indices to be sorted
// 将每个3D gaussian的对应的tile index和深度存到point_list_keys_unsorted中
// 将每个3D gaussian的对应的index（第几个3D gaussian）存到point_list_unsorted中
    duplicateWithKeys << <(P + 255) / 256, 256 >> > (   // 根据 tile，复制 Gaussian
            P,
                    geomState.means2D,  // 预处理计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
                    geomState.depths,   // 预处理计算的 所有高斯 中心在当前相机坐标系下的z值 数组
                    geomState.point_offsets,    //
                    binningState.point_list_keys_unsorted,
                    binningState.point_list_unsorted,
                    radii,          // 预处理计算的 所有高斯 投影在当前相机图像平面的最大半径 数组
                    tile_grid)      // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
    CHECK_CUDA(, debug)