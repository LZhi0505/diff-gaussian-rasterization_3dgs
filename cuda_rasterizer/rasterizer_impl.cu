/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>      // CUDA的CUB库
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>      // GLM (OpenGL Mathematics)库

#include <cooperative_groups.h>     // CUDA 9引入的Cooperative Groups库
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

/**
 * 引用库的介绍
 * 1. cooperative_groups库（同步）
 * __syncthreads()函数提供了在一个 block内同步各线程的方法，但有时要同步 block内的一部分线程或者多个 block的线程，这时候就需要 Cooperative Groups库。这个库定义了划分和同步一组线程的方法
 * 在3DGS中方法仅以两种方式被调用：
 * (1) auto idx = cg::this_grid().thread_rank();    其中 cg::this_grid()返回一个 cg::grid_group实例，表示当前线程所处的 grid。它有一个方法 thread_rank()返回当前线程在该 grid中排第几
 * (2) auto block = cg::this_thread_block();    其中 cg::this_thread_block返回一个 cg::thread_block实例，表示当前线程所处的 block，用到的成员函数有：
 *      block.sync()：同步该 block中的所有线程（等价于__syncthreads()）
 *      block.thread_rank()：返回非负整数，表示当前线程在该 block中排第几
 *      block.group_index()：返回一个 cg::dim3实例，表示该 block在 grid中的三维索引
 *      block.thread_index()：返回一个 cg::dim3实例，表示当前线程在 block中的三维索引
 *
 * 2. CUB库（并行处理）
 * 针对不同的计算等级：线程、wap、block、device等设计了并行算法。例如，reduce函数有四个版本：ThreadReduce、WarpReduce、BlockReduce、DeviceReduce
 * diff-gaussian-rasterization模块调用了CUB库的两个函数：
 * (1) cub::DeviceScan::InclusiveSum    计算前缀和，'InclusiveSum'是从第一个元素 累加到 当前元素 的和
 * (2) cub::DeviceRadixSort::SortPairs  device级别的并行基数 升序排序
 *
 * 3. GLM库
 * 专为图形学设计的只有头文件的C++数学库
 * 3DGS只用到了 glm::vec3（三维向量）, glm::vec4（四维向量）, glm::mat3（3×3矩阵）, glm::dot（向量点积）
 */


/**
 * 计算 tile总数的 二进制数中的 最高有效位 MSB的位置（二分法），用于确定位操作的范围
 * @param n CUDA网格的 tile总数
 */
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;   // 初值设为 n位数的一半，4 * 4 = 16 bit
	uint32_t step = msb;    // 初始步长
	while (step > 1)
	{
		step /= 2;      // 步长缩小一半
		if (n >> msb)   // 如果 n右移 msb位后不为 0，说明最高有效位在更高的位置
			msb += step;
		else            // 如果 n右移 msb位后为 0，说明最高有效位在更低的位置
			msb -= step;
	}
	if (n >> msb)   // 确保 msb是最高有效位的实际位置
		msb++;
	return msb;
}



/**
 * 检查某个线程处理的高斯是否在当前相机的视锥体内
 */
__global__ void checkFrustum(
    int P,          // 所有高斯的个数
	const float* orig_points,   // 所有高斯 中心的世界坐标
	const float* viewmatrix,    // 观测变换矩阵，W2C
	const float* projmatrix,    // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	bool* present)      // 输出的 所有高斯 是否在当前相机视锥体内的标志 数组
{
    // 获取当前线程处理的高斯的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;  // 该高斯在当前相机坐标系下的坐标
    // 检查该高斯是否在当前相机的视锥体内
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


/**
 * 为每个高斯覆盖的所有 tile生成用于排序的 key-value，以便在后续操作中按深度对高斯进行排序
 */
__global__ void duplicateWithKeys(
	int P,      // 所有高斯的个数
	const float2* points_xy,    // 预处理计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
	const float* depths,        // 预处理计算的 所有高斯 中心在当前相机坐标系下的z值（深度）数组
	const uint32_t* offsets,    // 所有高斯 覆盖的 tile个数的 前缀和 数组
	uint64_t* gaussian_keys_unsorted,   // 输出的 遍历所有高斯生成它们覆盖的tile的 且 未排序的 keys 列表。每个元素64bit，高32位存某高斯覆盖的tile_ID，低32位存某高斯中心在相机坐标系下的z（深度）值
	uint32_t* gaussian_values_unsorted, // 输出的 遍历所有高斯生成它们覆盖的tile的 且 未排序的 values 列表。每个元素是 某高斯的ID
	int* radii,     // 预处理计算的 所有高斯 投影在当前相机图像平面的最大半径 数组
	dim3 grid)      // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
{
    // 获取当前线程处理的高斯的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    // 只有该3D高斯投影到当前相机的图像平面的最大半径 > 0，即当前相机看见了该高斯，才生成 key-value
	if (radii[idx] > 0)
	{
        // 该高斯 前面的那些高斯已经覆盖的 tile的总数，即前一个高斯覆盖的tile的终止位置，也是该高斯覆盖的tile的起始位置
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        // 计算该高斯投影到当前相机图像平面的 覆盖区域的左上角和右下角 tile块坐标
		uint2 rect_min, rect_max;
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // 遍历该高斯 覆盖的每个 tile，为其生成一个 key-value：tile_ID|高斯深度 - 该高斯的ID
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;  // tile在整幅图像的 ID
				key <<= 32;         // 高位存 tile ID
				key |= *((uint32_t*) & depths[idx]);      // 低位存 该3D高斯在当前相机坐标系下的 深度

                // 为该高斯覆盖的当前 tile 分配 key-value
                gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;      // tile数组中的偏移量
			}
		}
	}
}


/**
 * 通过遍历排序后的 point_list_keys 列表，为每个 tile 计算出它在整个 point_list_keys 列表中的起始和终止位置，并将这些位置存储到 ranges 数组中
 */
__global__ void identifyTileRanges(
        int L,      // 排序的 tile总个数，即所有高斯 投影到二维图像平面上覆盖的 tile的总个数
        uint64_t* point_list_keys,  // 根据tile ID和高斯深度排序后的 keys列表
        uint2* ranges)  // ranges[tile_ID].x 和 y 表示 第 tile_ID个 tile在排过序的keys列表中的起始和终止位置
{
    // 获取当前线程的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

    // 读取 当前线程处理的 key，[tile ID | 深度]
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;  // 当前tile的ID
	if (idx == 0)
        // 如果是第一个 tile，则其起始位置在索引 0 处
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        // 当前tile 和 前一个tile 不同，则记录前一个tile的终止位置和当前tile的起始位置
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;   // 前一个 tile的终止位置是当前索引 idx
			ranges[currtile].x = idx;   // 当前 tile的起始位置也是 idx
		}
	}
	if (idx == L - 1)
        // 如果是最后一个 tile，则其终止位置在索引 L 处
		ranges[currtile].y = L;
}


/**
 * 检查所有高斯是否在当前相机的视锥体内
 */
void CudaRasterizer::Rasterizer::markVisible(
	int P,          // 所有高斯的个数
	float* means3D,     // 所有高斯 中心的世界坐标
	float* viewmatrix,  // 观测变换矩阵，W2C
	float* projmatrix,  // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	bool* present)      // 输出的 所有高斯 是否在当前相机视锥体内的标志 数组
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// CUDA内存状态类，用于在GPU内存中存储和管理不同类型的数据
// fromChunk：从以 char数组形式存储的二进制块中读取 GeometryState、ImageState、BinningState等类的信息
/**
 * (1) 存储与高斯几何相关的信息，从动态分配的内存块(char*& chunk)中 提取并初始化 GeometryState结构（与高斯各参数的数据成员）
 * 使用 obtain 函数为 GeometryState 的不同成员分配空间，并返回一个初始化的 GeometryState 实例
 * @param chunk 一个指向内存块的指针引用
 * @param P     所有高斯的个数
 */
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);         // 所有高斯 中心在当前相机坐标系下的z值（深度） 数组
	obtain(chunk, geom.clamped, P * 3, 128);    // 所有高斯 是否被裁剪的标志 数组
	obtain(chunk, geom.internal_radii, P, 128); // 所有高斯 在图像平面上的投影半径
	obtain(chunk, geom.means2D, P, 128);        // 所有高斯 中心投影在当前相机图像平面的二维坐标 数组
	obtain(chunk, geom.cov3D, P * 6, 128);  // 所有高斯 在世界坐标系下的3D协方差矩阵 数组
	obtain(chunk, geom.conic_opacity, P, 128);  // 所有高斯 2D协方差的逆 和 不透明度 数组
	obtain(chunk, geom.rgb, P * 3, 128);    // 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
	obtain(chunk, geom.tiles_touched, P, 128);  // 所有高斯 在当前相机图像平面覆盖的线程块 tile的个数 数组

    // 计算前缀和，InclusiveSum表示包括自身，ExclusiveSum表示不包括自身
    // 当临时所需的显存空间为 NULL时，所需的分配空间大小被写入到 第二个参数中，并且不执行任何操作
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);

    obtain(chunk, geom.scanning_space, geom.scan_size, 128);    // 用于计算前缀和的中间缓冲区，数据的对齐方式为 128字节
	obtain(chunk, geom.point_offsets, P, 128);  // 所有高斯 覆盖的 tile个数的 前缀和 数组，每个元素是 从第一个高斯到当前高斯所覆盖的所有tile的数量
	return geom;
}

/**
 * (2) 给 ImageState img分配所需的内存
 * @param N 图片中 tile的总数
 */
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128); // 渲染后每个像素 pixel的 累积的透射率 的数组
	obtain(chunk, img.n_contrib, N, 128);   // 渲染每个像素 pixel穿过的高斯的个数，也是最后一个对渲染该像素RGB值 有贡献的高斯ID 的数组
	obtain(chunk, img.ranges, N, 128);      // 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile_ID；值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数
	return img;
}

/**
 * (3) 给 BinningState binning分配所需的内存，并执行排序操作
 */
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);  // 排序后的 value列表
	obtain(chunk, binning.point_list_unsorted, P, 128); // 未排序的 所有高斯覆盖的tile的 values列表，每个元素是 对应高斯的ID

    obtain(chunk, binning.point_list_keys, P, 128);     // 排序后的 keys列表，分布顺序：大顺序：各tile_ID，小顺序：落在该tile内各高斯的深度
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);    // 未排序的 所有高斯覆盖的tile的 keys列表，分布顺序：大顺序：各高斯，小顺序：该高斯覆盖的各tile_ID。每个元素是 (tile_ID | 3D高斯的深度)

    // GPU上device级别的并行基数 升序排序, 将 point_list_keys_unsorted 作为键，point_list_unsorted 作为值进行排序，排序结果存储在 point_list_keys 和 point_list 中
    cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);

    // list_sorting_space 用于排序的临时空间
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}


/**
 * 可微光栅化的 前向传播，可当作 main 函数
 * 1: 分配 (p+255/256)个 block，每个 block有 256个 thread，对每个高斯做 preprocessCUDA();
 * 2: 生成 buffer并对高斯做排序；
 * 3: 分配 num_tiles个 block，每个 block有 256个 thread，对每个 pixel做渲染
 * return：num_rendered：所有高斯 投影到二维图像平面上覆盖的 tile的总个数 = 排序列表的长度
 */
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
	float* out_color,       // 输出的 RGB图像，考虑了背景颜色，(3,H,W)
	int* radii,             // 输出的 所有高斯 投影在当前相机图像平面的最大半径 数组，(N,)
	bool debug)     // 默认为False
{
    // 1. 计算焦距，W = 2fx * tan(Fovx/2) ==> fx = W / (2 * tan(Fovx/2))
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

    // 2. 为 GeometryState分配显存，每个高斯都有一个 GeometryState数据
	size_t chunk_size = required<GeometryState>(P);     // 根据高斯的数量 P，模版函数 required调用 fromChunk函数来获取内存，返回结束地址，也即所需的存储空间大小
	char* chunkptr = geometryBuffer(chunk_size);        // 根据所需的存储空间大小，调用 rasterize_points.cu文件中的 resizeFunctional函数里面嵌套的匿名函数 lambda来调整显存块大小，并返回首地址 chunkptr
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);    // 用显存块首地址作为参数，调用 fromChunk函数为 GeometryState geo申请显存

	if (radii == nullptr) {
        // 如果传入的、要输出的 高斯在图像平面的投影半径 为空指针，则将其设为 geomState缓存的投影半径
		radii = geomState.internal_radii;
	}

    // 3. 定义一个 tile_grid的维度，即在水平和垂直方向上需要多少个线程块 block来覆盖整个渲染区域，(W/16, H/16, 1)
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    // 定义一个 block的维度，即在水平和垂直方向上的线程 thread个数。每个线程处理一个像素，则每个block处理16*16个像素，(16, 16, 1)
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 4. 为 ImageState分配显存，每个像素都有一个 ImageState数据
	size_t img_chunk_size = required<ImageState>(width * height);   // 计算存储所有2D像素各参数 所需的空间大小
	char* img_chunkptr = imageBuffer(img_chunk_size);                  // 分配存储空间, 并返回指向该存储空间的指针
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);  // 在给定的内存块中初始化 ImageState 结构体, 为不同成员分配空间，并返回一个初始化的实例

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

    //! 5. 预处理和投影。具体实现在 forward.cu/preprocessCUDA
    // (1) 将每个高斯投影到图像平面上，计算2D协方差矩阵、投影半径 radii；
    // (2) 计算投影所占的tile块坐标和个数 tile tiles_touched；
    // (3) 如果用球谐系数，将其转换成RGB；
    // (4) 记录高斯的像素坐标 points_xy_image
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,      // 输出的 所有高斯 是否被裁剪的标志 数组，某位置为True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
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
		geomState.tiles_touched,    // 输出的 所有高斯 在当前相机图像平面覆盖的线程块 tile的个数 数组
		prefiltered                 // 预滤除的标志，默认为False
	), debug)

    //! 6. 高斯排序：根据高斯距离摄像机的远近来并行计算 每个高斯在 α-blending中的顺序
    // 在GPU上并行计算 每个高斯投影到当前相机图像平面上 2D高斯覆盖的 tile个数的 前缀和，结果存储在 point_offsets，提供了每个高斯覆盖tile区域的累加结束位置
    // 是为 所有高斯投影到图像平面上覆盖的所有 tile分配唯一的 ID
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space,  // 额外需要的临时显存空间
                                             geomState.scan_size,       // 临时显存空间的大小
                                             geomState.tiles_touched,   // 输入指针，已计算的 每个高斯 投影到当前相机图像平面覆盖的 tile个数的 数组
                                             geomState.point_offsets,   // 输出指针，指向一个数组，每个元素是 从第一个高斯到当前高斯所覆盖的所有 tile的 数量
                                             P      // 所有高斯的个数
                                             ), debug)

    // 计算所有高斯 投影到二维图像平面上覆盖的 tile的总个数 = 排序列表的长度
	int num_rendered;
    // 将 point_offsets数组的最后一个元素，即所有高斯投影到当前相机图像平面上所覆盖的 tile的 总数，从GPU复制到CPU的变量 num_rendered中
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    // 为 BinningState分配显存，即每个高斯覆盖的 tile都有一个 BinningState数据，其存储着 排序前和排序后的 key、value列表
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // 生成排序列表：对于每个要渲染的高斯，遍历其覆盖的tile，生成排序前的keys列表和values列表
    // point_list_keys_unsorted:    未排序的keys列表，分布顺序：大顺序：各高斯，小顺序：该高斯覆盖的各tile_ID。每个元素64bit，高32位存某高斯覆盖的tile_ID，低32位存某高斯中心在相机坐标系下的z（深度）值，即(tile_ID | 3D高斯的深度)
    // point_list_unsorted:         未排序的values列表，每个元素是 对应高斯的ID
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,  // 预处理计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
		geomState.depths,   // 预处理计算的 所有高斯 中心在当前相机坐标系下的z值（深度） 数组
		geomState.point_offsets,    // 所有高斯覆盖的 tile个数的 前缀和
		binningState.point_list_keys_unsorted,  // 输出的 未排序的 遍历所有高斯生成它们覆盖的tile的 keys 列表，每个元素是 (tile_ID | 3D高斯的深度)
		binningState.point_list_unsorted,       // 输出的 未排序的 遍历所有高斯生成它们覆盖的tile的 values 列表，每个元素是 对应高斯的ID
		radii,          // 预处理计算的 所有高斯 投影在当前相机图像平面的最大半径 数组
		tile_grid)      // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
	CHECK_CUDA(, debug)

    // 计算 tile总数的 二进制数中的 最高有效位的 位置，用于确定位操作的范围
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);


    // 排序：对于每个tile，遍历落在其上的高斯，按各高斯的深度升序排序，生成排序后的keys列表和values列表
    // 按 key的大小，即tile_ID和3D高斯的深度，对 keys、values列表进行稳定的、并行、基数 升序排序
    // point_list_keys: 排序后的keys列表，分布顺序：大顺序：各tile_ID，小顺序：落在该tile内各高斯的深度
    // point_list:      排序后的values列表
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,    // 排序时用到的临时显存空间
		binningState.sorting_size,                     // 临时显存空间的大小
		binningState.point_list_keys_unsorted,  // 未排序的 每个高斯覆盖的所有 tile的 keys列表。 大顺序：各高斯，小顺序：该高斯覆盖的各tile_ID
        binningState.point_list_keys,           // 排序后的 keys列表。                       大顺序：各tile_ID，小顺序：落在该tile内各高斯的深度
		binningState.point_list_unsorted,   // 未排序的 每个高斯覆盖的所有 tile的 values列表。每个元素是[对应3D高斯的 ID]
        binningState.point_list,            // 排序后的 values列表
		num_rendered,   // 要排序的 tile总个数，即所有高斯 投影到二维图像平面上覆盖的 tile的总个数
        0,      // 指定时从最低位开始
        32 + bit    // 指定排序的最高位，表示排序的范围是从第 0位到第 32 + bit位。bit代表了 tile ID的最高位数。加上 32 是因为 tile ID和深度值分别占据了32位
        ), debug)

    // 将CUDA设备内存中的一块区域 imgState.ranges 数组中的所有元素初始化为 0（uint2是一个由两个 uint32_t组成的结构体，所以其大小是 8字节）
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);


    // 根据排序后的keys列表，为每个 tile 计算 其在排序后的keys列表中的起始和终止位置，后续的渲染或处理步骤可以根据 tile ID 快速找到这个 tile 对应的高斯对象，而不需要再次进行复杂的查找或遍历
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,       // 排序的 tile总个数，即所有高斯 投影到二维图像平面上覆盖的 tile的总个数
			binningState.point_list_keys,   // 根据tile ID和高斯深度排序后的 keys列表
			imgState.ranges);   // 输出的 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile_ID；值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数。也可以用[x,y)在排序后的values列表中索引到该tile触及的所有高斯ID
	CHECK_CUDA(, debug)

    // 如果传入的预计算的颜色 不是空指针，则是预计算的颜色
    //                    是空指针（默认），则是 preprocess中计算的 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

    //! 7. 渲染: 在一个block上协作渲染一个tile内各像素的RGB颜色值，每个线程负责一个像素。具体实现在 forward.cu/renderCUDA
	CHECK_CUDA(FORWARD::render(
		tile_grid,     // 定义的CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数，(W/16，H/16)
        block,              // 定义的线程块 block的维度，(16, 16, 1)
		imgState.ranges,    // 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile ID，值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数。也可以用[x,y)在排序后的values列表中索引到该tile触及的所有高斯ID
		binningState.point_list,    // 按 tile ID、高斯深度 排序后的 values列表，即 高斯ID 列表
		width, height,
		geomState.means2D,  // 已计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
		feature_ptr,        // 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组（每个高斯在当前观测方向下只有一个颜色，仅颜色强度分布不同）
		geomState.conic_opacity,    // 已计算的 所有高斯 2D协方差矩阵的逆 和 不透明度 数组
		imgState.accum_alpha,   // 输出的 渲染后每个像素 pixel的 累积的透射率 的数组
		imgState.n_contrib,     // 输出的 渲染每个像素 pixel穿过的高斯的个数，也是最后一个对渲染该像素RGB值 有贡献的高斯ID 的数组
		background,     // 背景颜色，默认为[1,1,1]，黑色
		out_color               // 输出的 RGB图像（加上了背景颜色）
        ), debug)

	return num_rendered;
}


/**
 * 反向传播，计算loss对前向传播输出tensor的 梯度
 */
void CudaRasterizer::Rasterizer::backward(
	const int P,    // 所有高斯的个数
    int D,  // 当前的球谐阶数
    int M,  // 每个高斯的球谐系数个数=16
    int R,  // 所有高斯覆盖的 tile的总个数
	const float* background,    // 背景颜色，默认为[1,1,1]，黑色
	const int width, int height,
	const float* means3D,   // 所有高斯 中心的世界坐标
	const float* shs,       // 所有高斯的 球谐系数，(N,16,3)
	const float* colors_precomp,    // python代码中 预计算的颜色，默认是空tensor
	const float* scales,    // 所有高斯的 缩放因子
	const float scale_modifier, // 缩放因子调节系数
	const float* rotations, // 所有高斯的 旋转四元数
	const float* cov3D_precomp, // python代码中 预计算的3D协方差矩阵，默认为空tensor
	const float* viewmatrix,    // 观测变换矩阵，W2C
	const float* projmatrix,    // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	const float* campos,    // 当前相机中心的世界坐标
	const float tan_fovx, float tan_fovy,
	const int* radii,   // 所有高斯 投影在当前相机图像平面上的最大半径 数组
	char* geom_buffer,  // 存储所有高斯 几何数据的 tensor：包括2D中心像素坐标、相机坐标系下的深度、3D协方差矩阵等
	char* binning_buffer,   // 存储所有高斯 排序数据的 tensor：包括未排序和排序后的 所有高斯覆盖的tile的 keys、values列表
	char* img_buffer,       // 存储所有高斯 渲染后数据的 tensor：包括累积的透射率、最后一个贡献的高斯ID
	const float* dL_dpix,   // 输入的 loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
	float* dL_dmean2D,  // 输出的 loss对所有高斯 中心投影到图像平面的像素坐标的 导数
	float* dL_dconic,   // 输出的 loss对所有高斯 2D协方差矩阵的 导数
	float* dL_dopacity, // 输出的 loss对所有高斯 不透明度的 导数
	float* dL_dcolor,   // 输出的 loss对所有高斯 在当前相机中心的观测方向下 的RGB颜色值 导数
	float* dL_dmean3D,  // 输出的 loss对所有高斯 中心世界坐标的 导数
	float* dL_dcov3D,   // 输出的 loss对所有高斯 3D协方差矩阵的 导数
	float* dL_dsh,      // 输出的 loss对所有高斯 球谐系数的 导数
	float* dL_dscale,   // 输出的 loss对所有高斯 缩放因子的 导数
	float* dL_drot,     // 输出的 loss对所有高斯 旋转四元数的 导数
	bool debug) // 默认为False
{
    // 这些缓冲区都是在前向传播的时候存下来的，现在拿出来用
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr) {
        // 如果传入的 所有高斯投影在当前相机图像平面上的最大半径数组为 空指针，则从geomState缓冲区中获取
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1); // 线程块 block（tile）的维度，(W/16, H/16, 1)
	const dim3 block(BLOCK_X, BLOCK_Y, 1);  // 一个block中 线程thread的维度，(16, 16, 1)

    // 如果传入的预计算的颜色 不是空指针，则是预计算的颜色
    //                    是空指针（默认），则是 preprocess中计算的 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;

    //! 反传中的 渲染 部分：计算 loss对 所有高斯的2D中心坐标、圆锥矩阵、不透明度和RGB等数据的 导数。具体实现在 backward.cu/renderCUDA
	CHECK_CUDA(BACKWARD::render(
		tile_grid,  // 线程块 block（tile）的维度，(W/16, H/16, 1)
		block,  // 一个block中 线程thread的维度，(16, 16, 1)
		imgState.ranges,    // 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile_ID；值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数
		binningState.point_list,    // 按深度排序后的 所有高斯覆盖的tile的 values列表，每个元素是 对应高斯的ID
		width, height,
		background,     // 背景颜色，默认为[1,1,1]，黑色
		geomState.means2D,      // 所有高斯 中心投影在当前相机图像平面的二维坐标 数组
		geomState.conic_opacity,    // 所有高斯 2D协方差的逆 和 不透明度 数组
		color_ptr,      // 默认是 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
		imgState.accum_alpha,   // 渲染后每个像素 pixel的 累积的透射率 的数组
		imgState.n_contrib,     // 渲染每个像素 pixel穿过的高斯的个数，也是最后一个对渲染该像素RGB值 有贡献的高斯ID 的数组
		dL_dpix,    // 输入的 loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
		(float3*)dL_dmean2D,    // 输出的 loss对所有高斯 中心投影到图像平面的像素坐标的 导数
		(float4*)dL_dconic,     // 输出的 loss对所有高斯 2D协方差矩阵的 导数
		dL_dopacity,            // 输出的 loss对所有高斯 不透明度的 导数
		dL_dcolor),     // 输出的 loss对所有高斯 在当前相机中心的观测方向下 的RGB颜色值 导数
        debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
    // 处理预处理的剩余部分
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;

    //! 反传中的 预处理 部分：
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}