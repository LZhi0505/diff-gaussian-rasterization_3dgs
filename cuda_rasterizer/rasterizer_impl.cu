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
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


// Helper function to find the next-highest bit of the MSB on the CPU.
// 寻找给定无符号整数 n 的最高有效位（Most Significant Bit, MSB）的下一个最高位
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

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // 对于边界矩形重叠的每个瓦片，具有一个键/值对。
        // 键是 | 瓦片ID | 深度 |，
        // 值是高斯的ID，按照这个键对值进行排序，将得到一个高斯ID列表，
        // 这样它们首先按瓦片排序，然后按深度排序
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
// 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// CUDA内存状态类，用于在GPU内存中存储和管理不同类型的数据
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
		radii,              // 输出的 所有高斯 投影在图像平面的最大半径 数组
		geomState.means2D,  // 输出的 所有高斯 中心在图像平面的二维坐标 数组
		geomState.depths,   // 输出的 所有高斯 中心在相机坐标系下的z值 数组
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
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
    // 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在 binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小
    // num_rendered 是要排序的元素总数。0, 32 + bit 指定了排序的最低位和最高位，这里用于确保排序考虑到了足够的位数，以便正确处理所有的键值对
    // Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

    // 将 imgState.ranges 数组中的所有元素设置为 0
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
    // 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
    // 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (   // 根据有序的Gaussian列表，判断每个 tile 需要跟哪一个 range 内的 Gaussians 进行计算
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
    // ---结束--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

    //! 3. 渲染
    // 具体实现在 forward.cu/renderCUDA
	CHECK_CUDA(FORWARD::render(
		tile_grid,     // 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
        block,              // 每个块在 X（水平）和 Y（垂直）方向上的线程数
		imgState.ranges,    // 每个瓦片（tile）在排序后的高斯ID列表中的范围
		binningState.point_list,    // 排序后的3D gaussian的id列表
		width, height,      // 图像的宽和高
		geomState.means2D,  // 每个2D高斯在图像上的中心点位置
		feature_ptr,        // 每个3D高斯对应的RGB颜色
		geomState.conic_opacity,    // 每个2D高斯的协方差矩阵的逆矩阵以及它的不透明度
		imgState.accum_alpha,   // 渲染过程后每个像素的最终透明度或透射率值
		imgState.n_contrib,     // 每个pixel的最后一个贡献的2D gaussian是谁
		background,     // 背景颜色
		out_color), debug)      // 输出图像

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