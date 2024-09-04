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
 * (1) cub::DeviceScan::InclusiveSum    计算前缀和，"Inclusive"就是第 i 个数被计入第 i 个和中
 * (2) cub::DeviceRadixSort::SortPairs  device级别的并行基数 升序排序
 *
 * 3. GLM库
 * 专为图形学设计的只有头文件的C++数学库
 * 3DGS只用到了 glm::vec3（三维向量）, glm::vec4（四维向量）, glm::mat3（3×3矩阵）, glm::dot（向量点积）
 */


// 在CPU上查找 给定无符号整数 n 的最高有效位（Most Significant Bit, MSB）的下一个最高位
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}


// 检查某个线程的高斯是否在当前相机的视锥体内，bool类型输出到 present数组内
__global__ void checkFrustum(
    int P,          // 所有高斯的个数
	const float* orig_points,   // 所有高斯 中心的世界坐标
	const float* viewmatrix,    // 观测变换矩阵，W2C
	const float* projmatrix,    // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	bool* present)      // 输出的 所有高斯是否被当前相机看见的标志
{
    // 获取当前线程处理的高斯的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;  // 输出的 该高斯在相机坐标系下的位置
    // 检查，如果不在当前相机视锥体内，则为False
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}


/**
 * 为每个高斯生成 用于排序的 [key | value]对，以便在后续操作中按深度对高斯进行排序，为每个高斯运行一次（1:N 映射）
 * key：每个tile对应的深度（depth）
 * value: 对应的高斯索引
 */
__global__ void duplicateWithKeys(
	int P,      // 所有高斯的个数
	const float2* points_xy,    // 预处理计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
	const float* depths,        // 预处理计算的 所有高斯 中心在当前相机坐标系下的z值 数组
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,   // 输出的 未排序的 keys
	uint32_t* gaussian_values_unsorted, // 输出的 未排序的 values
	int* radii,     // 预处理计算的 所有高斯 投影在当前相机图像平面的最大半径 数组
	dim3 grid)      // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
{
    // 获取当前线程处理的高斯的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

    // 当前相机看不见某高斯，则不生成[key | value]对
	if (radii[idx] > 0)
	{
        // 寻找该高斯在缓冲区的 offset，以生成[key | value]对
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        //
		uint2 rect_min, rect_max;
        // 计算该高斯投影在当前相机图像平面的影响范围（左上角和右下角坐标的 线程块坐标），以给高斯覆盖的每个tile生成一个(key, value)对
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // 对于边界矩形重叠的每个瓦片，具有一个键/值对。
        // 键是 | 瓦片ID | 深度 |，
        // 值是高斯的ID，按照这个键对值进行排序，将得到一个高斯ID列表，
        // 这样它们首先按瓦片排序，然后按深度排序
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;  // tile的ID
				key <<= 32;         // 放在高位
				key |= *((uint32_t*)&depths[idx]);      // 低位是深度
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;      // 数组中的偏移量
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
// 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
__global__ void identifyTileRanges(
        int L,      // 排序列表中的元素个数
        uint64_t* point_list_keys,  // 排过序的keys
        uint2* ranges)  // ranges[tile_id].x和y表示第tile_id个tile在排过序的列表中的起始和终止地址
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;  // 当前tile
	if (idx == 0)
		ranges[currtile].x = 0;     // 边界条件：tile 0的起始位置
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
            // 上一个元素和我处于不同的tile，
            // 那我是上一个tile的终止位置和我所在tile的起始位置
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;     // 边界条件：最后一个tile的终止位置
}

// 检查所有高斯是否被在当前相机的视锥体内，即是否被当前相机看见，标志保存在 present数组中
void CudaRasterizer::Rasterizer::markVisible(
	int P,          // 所有高斯的个数
	float* means3D,     // 所有高斯 中心的世界坐标
	float* viewmatrix,  // 观测变换矩阵，W2C
	float* projmatrix,  // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	bool* present)      // 输出的 所有高斯是否被当前相机看见的标志
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
	obtain(chunk, geom.depths, P, 128);         // 所有高斯 在相机坐标系下的深度
	obtain(chunk, geom.clamped, P * 3, 128);    // 所有高斯 是否被裁剪的标志
	obtain(chunk, geom.internal_radii, P, 128); // 所有高斯 在图像平面上的投影半径
	obtain(chunk, geom.means2D, P, 128);    // 输出的 所有高斯 中心投影到图像平面的坐标
	obtain(chunk, geom.cov3D, P * 6, 128);  // 所有高斯 在世界坐标系下的3D协方差矩阵
	obtain(chunk, geom.conic_opacity, P, 128);  // 所有高斯的 2D协方差的逆、不透明度
	obtain(chunk, geom.rgb, P * 3, 128);    // 所有高斯的 RGB颜色
	obtain(chunk, geom.tiles_touched, P, 128);  // 所有高斯 覆盖的 tile数量

	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);

    obtain(chunk, geom.scanning_space, geom.scan_size, 128);    // 用于计算前缀和的中间缓冲区，数据的对齐方式为 128字节
	obtain(chunk, geom.point_offsets, P, 128);  // 每个高斯在有序列表中的位置
	return geom;
}

// (2) 存储与图像渲染相关的信息
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128); // 每个像素的累积 alpha值
	obtain(chunk, img.n_contrib, N, 128);   // 每个像素的贡献高斯数量
	obtain(chunk, img.ranges, N, 128);      // 每个 tile 所需的高斯范围
	return img;
}

/**
 * (3) 初始化 BinningState 实例，分配所需的内存，并执行排序操作
 */
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);  // 排序后的高斯分布索引列表
	obtain(chunk, binning.point_list_unsorted, P, 128); // 未排序的高斯分布索引列表
	obtain(chunk, binning.point_list_keys, P, 128);     // 排序后的 (tile, depth) 键列表
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);    // 未排序的 (tile, depth) 键列表

    // 在 GPU 上进行基数排序, 将 point_list_keys_unsorted 作为键，point_list_unsorted 作为值进行排序，排序结果存储在 point_list_keys 和 point_list 中
    cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);

    // list_sorting_space 用于排序的临时空间
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}


/**
 * 高斯的可微光栅化的前向渲染处理，可当作 main 函数
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
	float* out_color,       // 输出的 颜色图像，(3,H,W)
	int* radii,             // 输出的 在图像平面上的投影半径(N,)
	bool debug)     // 默认为False
{
    // 1. 计算焦距，W = 2fx * tan(Fovx/2) ==> fx = W / (2 * tan(Fovx/2))
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

    // 2. 分配、初始化几何信息 geomState
	size_t chunk_size = required<GeometryState>(P);     // 根据高斯的数量 P，计算存储所有高斯各参数 所需的空间大小
	char* chunkptr = geometryBuffer(chunk_size);        // 分配指定大小 chunk_size的缓冲区，即给所有高斯的各参数分配存储空间，返回指向该存储空间的指针 chunkptr
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);    // 在给定的内存块中初始化 GeometryState 结构体, 为不同成员分配空间，并返回一个初始化的实例

	if (radii == nullptr) {
        // 如果传入的、要输出的 高斯在图像平面的投影半径为 nullptr，则将其设为
		radii = geomState.internal_radii;
	}

    // 3. 定义一个 tile_grid的大小，即在水平和垂直方向上需要多少个线程块来覆盖整个渲染区域，W/16，H/16
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    // 定义一个 block的大小，即在水平和垂直方向上的线程数。每个线程处理一个像素，则每个线程块处理16*16个像素
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 4. 分配、初始化图像信息 ImageState
	size_t img_chunk_size = required<ImageState>(width * height);   // 计算存储所有2D像素各参数 所需的空间大小
	char* img_chunkptr = imageBuffer(img_chunk_size);                  // 分配存储空间, 并返回指向该存储空间的指针
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);  // 在给定的内存块中初始化 ImageState 结构体, 为不同成员分配空间，并返回一个初始化的实例

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

    //! 5. 预处理和投影：将每个高斯投影到图像平面上、计算投影所占的tile块坐标和个数、根据球谐系数计算RGB值。 具体实现在 forward.cu/preprocessCUDA
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
		geomState.tiles_touched,    // 输出的 所有高斯 在当前相机图像平面覆盖的线程块 tile的个数 数组
		prefiltered                 // 预滤除的标志，默认为False
	), debug)

    //! 6. 高斯排序和合成顺序：根据高斯距离摄像机的远近来计算每个高斯在Alpha合成中的顺序
    // ---开始--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    // 在GPU上并行计算每个高斯 投影到当前相机图像平面上 其投影圆覆盖的线程块 tile的个数的 前缀和，结果存储在 point_offsets中
    // 前缀和的目的：计算出每个高斯对应的 tile在数组中的起始位置（为 duplicateWithKeys做准备）
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space,  // 额外需要的临时显存空间
                                             geomState.scan_size,       // 临时显存空间的大小
                                             geomState.tiles_touched,   // 输入指针，已计算的 所有高斯 投影到当前相机图像平面覆盖的 tile个数的 数组
                                             geomState.point_offsets,   // 输出指针，每个高斯对应的 tile在数组中的起始位置
                                             P      // 元素个数
                                             ), debug)

    // 计算所有高斯 投影到二维图像平面上 总共覆盖的 tile的个数
	int num_rendered;
    // 从 GPU内存复制最后一个高斯的偏移量，得到需要渲染的高斯的总数
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    // 6.3 分配、初始化排序信息 BinningState，存储要 排序的[key | value]对 和 排序后的结果
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 对于每个要渲染的高斯, 生成 [ tile | depth ] key，和对应的 要排序的 高斯索引
    // 将每个高斯的对应的 tile index 和 深度存到 point_list_keys_unsorted中
    // 将每个3D gaussian的对应的index（第几个3D gaussian）存到point_list_unsorted中 // 生成排序所用的keys和values
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,  // 预处理计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
		geomState.depths,   // 预处理计算的 所有高斯 中心在当前相机坐标系下的z值 数组
		geomState.point_offsets,    // 所有高斯的偏移量数组
		binningState.point_list_keys_unsorted,  // 未排序的键
		binningState.point_list_unsorted,       // 未排序的值
		radii,          // 预处理计算的 所有高斯 投影在当前相机图像平面的最大半径 数组
		tile_grid)      // 生成排序所用的 keys和 values。CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);


    // 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在 binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小
    // num_rendered 是要排序的元素总数。0, 32 + bit 指定了排序的最低位和最高位，这里用于确保排序考虑到了足够的位数，以便正确处理所有的键值对
    // Sort complete list of (duplicated) Gaussian indices by keys
    // 进行排序，按keys的大小升序排序：每个 tile对应的多个高斯按深度放在一起；value是Gaussian的ID
    // cub::DeviceRadixSort::SortPairs：device级别的并行基数排序，根据 key的大小将 (key, value)对 进行稳定的升序排序
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,    // 排序时用到的临时显存空间
		binningState.sorting_size,                      // 临时显存空间的大小
		binningState.point_list_keys_unsorted, binningState.point_list_keys,    // key的输入和输出指针
		binningState.point_list_unsorted, binningState.point_list,  // value的输入和输出指针
		num_rendered,   // 对多少个条目进行排序
        0,          // 低位
        32 + bit    // 高位，按照[begin_bit, end_bit)内的位进行排序
        ), debug)

    // 将 imgState.ranges 数组中的所有元素设置为 0
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
    // 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
    // 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (   // 根据有序的Gaussian列表，判断每个 tile 需要跟哪一个 range 内的 Gaussians 进行计算
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);   // 计算每个tile对应排序过的数组中的哪一部分
	CHECK_CUDA(, debug)
    // ---结束--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

    //! 7. 渲染
    // 一个线程负责一个像素，一个block负责一个tile。线程在读取数据（把数据从公用显存拉到 block自己的显存）和进行计算之间来回切换，使得线程们可以共同读取高斯数据，这样做的原因是block共享内存比公共显存快得多。具体实现在 forward.cu/renderCUDA
	CHECK_CUDA(FORWARD::render(
		tile_grid,     // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
        block,              // 定义的线程块 block在水平和垂直方向上的线程数
		imgState.ranges,    // 每个线程块 tile在排序后的高斯ID列表中的范围
		binningState.point_list,    // 排序后的高斯的ID列表
		width, height,
		geomState.means2D,  // 已计算的 所有高斯 中心在当前相机图像平面的二维坐标 数组
		feature_ptr,        // 每个3D高斯对应的RGB颜色
		geomState.conic_opacity,    // 已计算的 所有高斯 2D协方差矩阵的逆 和 不透明度 数组
		imgState.accum_alpha,   // 渲染过程后 每个像素 pixel的最终透明度或透射率值
		imgState.n_contrib,     // 每个像素 pixel的最后一个贡献的高斯是谁
		background,     // 背景颜色，默认为[1,1,1]，黑色
		out_color), debug)      // 输出的 RGB图像

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding to forward render pass
// 产生对应于前向渲染过程所需的优化梯度
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
    // 根据每像素损失梯度计算损失梯度，关于2D均值位置、圆锥矩阵、
    // 高斯的不透明度和RGB。如果我们获得了预计算的颜色而不是球谐系数，就使用它们。
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    //! 核心渲染函数，定义在backward.h中，具体实现在 backward.cu/renderCUDA
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
    // 处理预处理的剩余部分
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
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