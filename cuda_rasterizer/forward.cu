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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/**
 * OpenGL Mathematics(glm)：是针对图形编程的数学库，用于OpenGL的开发，这个库基于C++的模板库。
 * 提供了各种数学功能和数据结构：
 * 向量（vec2, vec3, vec4）
 * 矩阵（mat2, mat3, mat4）
 * 四元数（quaterion）
 * 常见的数学函数（平移、旋转、缩放、透视投影等）
 */

/**
 * 核函数使用__global__修饰符声明。这表明这个函数是一个核函数，它可以在GPU上执行，而不能在CPU上执行。
 * 当调用一个核函数时，需要使用特殊的语法<<<grid, block>>>来指定执行配置。
 * 这个配置包括两个部分：
 * Grid：是指定了多少个块（block）组成的网格（grid），整个网格代表了所有并行执行单元的集合；
 * Block：是指每个块中包含多少个线程。块内的线程可以共享数据并协作执行任务。
 */

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
/**
 * 计算3D高斯的颜色
 * @param idx
 * @param deg 球谐函数的阶数
 * @param max_coeffs
 * @param means 所有高斯的中心位置
 * @param campos 相机中心位置
 * @param shs 球谐系数 (16, 3)
 * @param clamped
 * @return
 */
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// 该函数基于zhang等人的论文"Differentiable Point-Based Radiance Fields for Efficient View Synthesis"中的代码实现
	glm::vec3 pos = means[idx];		// 当前高斯的中心位置
	glm::vec3 dir = pos - campos;	// 从相机中心指向当前高斯的 方向
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;   // 获取当前高斯的球谐系数(16, 3)
	glm::vec3 result = SH_C0 * sh[0];                       // 计算0阶系数的颜色值
	// SH_C1等是球谐函数的基函数
	if (deg > 0) {
        // 阶数 > 0，则计算一阶SH系数的颜色值
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1) {
            // 阶数 > 1，则计算二阶SH系数的颜色值
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2) {
                // 阶数 > 2，则计算三阶SH系数的颜色值
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
    // 为结果颜色值加上一个偏移量
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
    // 将RGB颜色值限制在正值范围内。如果被限制，则需要在反向传播过程中记录此信息。
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
/**
 * 3DGS协方差矩阵 -> 2D的变换:
 * 1. 世界坐标系到相机坐标：viewmatirx
 * 2. 视锥到立方体：雅克比矩阵
 * @param mean      3D高斯中心的世界坐标
 * @param focal_x   相机在焦x轴方向上的焦距，也是视锥体近平面的深度
 * @param focal_y
 * @param tan_fovx
 * @param tan_fovy
 * @param cov3D     世界坐标系下的3G高斯的协方差矩阵
 * @param viewmatrix 观测变换矩阵，即世界坐标系 ==> 相机坐标系
 * @return 像素坐标系下的协方差矩阵，维度为(2,2)，但只返回了上半角元素(3个)
 */
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 雅克比矩阵在一个点附近才满足使用线性变换 近似 非线性变换的条件，而高斯的中心位置就是这个点，所以先求得3D高斯在相机坐标系下的位置）
	// 计算3D高斯中心在相机坐标系中的位置（在视锥中的位置）
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 构建雅克比矩阵（投影变换中的将视椎体压成立方体）
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// W：世界坐标系 ==> 相机坐标系的旋转矩阵 的转置
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    // 协方差矩阵从3D变为2D的公式：V_2D = JW V_3D W^T J^T
	glm::mat3 T = W * J;
    // 3D协方差矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 2D协方差矩阵 = J^T W V_3D^T W^T J
	// [sigma_x sigma_xy]
	// [sigma_xy sigma_y]
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// 进行低通滤波: 每个高斯应该在像素坐标系上的宽和高至少占1个像素上（忽略第三行和第三列）
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

/**
 * 前向传播中的方法：
 *     将每个3D高斯的旋转和缩放 转换为 世界坐标系下的3D协方差矩阵（需注意旋转四元数的归一化）
 * @param scale 缩放因子
 * @param mod   缩放因子调整系数
 * @param rot   旋转四元数
 * @param cov3D 输出的协方差矩阵
 */
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// 创建 缩放矩阵(3x3)
	glm::mat3 S = glm::mat3(1.0f);  // 初始化为一个3维的单位阵
	S[0][0] = mod * scale.x;    // 将缩放因子填入主对角线元素中
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// 将输入的四元数归一化 以正确表示 旋转
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// 四元数 => 旋转矩阵(3x3)
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// 计算世界坐标系下的 协方差矩阵(3x3)：R^T S^T S R
	glm::mat3 Sigma = glm::transpose(M) * M;

	// 因为协方差矩阵是对阵矩阵，因此只存储上半角元素
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}


//! 为每个3D高斯进行预处理的CUDA核函数
// 计算投影圆圈的半径：在3D空间中的高斯分布投影到2D图像平面时，它通常会形成一个圆圈（实际上是椭圆，因为视角的影响）。这个步骤涉及计算这个圆圈的半径。
// 计算圆圈覆盖的像素数：这涉及到将图像平面分成许多小块（tiles），并计算每个高斯分布投影形成的圆圈与哪些小块相交。这是为了高效地渲染，只更新受影响的小块。
template<int C>
__global__ void preprocessCUDA(
    int P,  // 3D高斯的数量
    int D,  // 3D高斯的维度
    int M,  // 点云数量
	const float* orig_points,   // 三维坐标
	const glm::vec3* scales,    // 缩放因子
	const float scale_modifier, // 缩放因子的调整系数
	const glm::vec4* rotations, // 旋转
	const float* opacities,     // 不透明
	const float* shs,           // 球谐系数
	bool* clamped,              // 用于记录是否被裁剪
	const float* cov3D_precomp, //预计算的3D协方差矩阵
	const float* colors_precomp,    // 预计算的颜色
	const float* viewmatrix,    // 观测变换矩阵
	const float* projmatrix,    // 观测变换矩阵 * 投影变换矩阵
	const glm::vec3* cam_pos,   // 相机位置
	const int W, int H,         // 输出图像的宽、高
	const float tan_fovx, float tan_fovy,   // 水平和垂直方向的视场角tan值
	const float focal_x, float focal_y,     // 焦距
	int* radii,                 // 输出的半径
	float2* points_xy_image,    // 输出的二维坐标
	float* depths,              // 输出的深度
	float* cov3Ds,              // 输出的3D协方差矩阵
	float* rgb,                 // 输出的颜色
	float4* conic_opacity,      // 锥形不透明度
	const dim3 grid,            // CUDA网格的大小
	uint32_t* tiles_touched,
	bool prefiltered)           // 是否预过滤
{
	auto idx = cg::this_grid().thread_rank();	// 获取当前线程对应的下标（一个线程处理一个像素）
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
    // 初始化半径、触及到的瓦片数量为0，如果这个值没有改变，说明这个高斯将不会进行后面的预处理
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
    // 给定指定的相机姿势，确定哪些3D高斯位于相机的视锥体之外。这样做可以确保在后续计算中不涉及给定视图之外的3D高斯，从而节省计算资源。
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        // 如果该3DGS不在视锥体内，则返回
		return;

	// Transform point by projecting
    // 将当前3D高斯idx投影到2D图像平面上
	// 1. 将3D高斯的 中心 从3D变换到2D：包含观测变换、投影变换、视口变换、光栅化
    // 1.1 该3D高斯中心的世界坐标
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// 1.2 中心：世界坐标系 ==> NDC坐标系，转换后该点处于[-1,1]的正方体中
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);   // projmatrix = 观测变换 * 投影变换
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
    // 1.3 归一化后的、在NDC坐标系中的高斯中心
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // 2. 将3D高斯的 协方差矩阵 从3D变换到2D：包含观测变换(viewmatrix)、投影变换中的视锥到立方体的变换(雅可比矩阵)
	// 2.1 获取3D协方差矩阵
	const float* cov3D;
	if (cov3D_precomp != nullptr) {
		// 如果提供了预计算的3D协方差矩阵，则直接使用它
		cov3D = cov3D_precomp + idx * 6;
	} else {
		// 默认未提供，则从缩放因子和旋转四元数中计算世界坐标系下的 3D协方差矩阵
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}
    // 2.2 协方差矩阵：世界坐标系 ==观测变换==> 相机坐标系 ==投影变换中视锥到立方体==> 立方体中
    // 2D协方差矩阵的上半角元素(3个)
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 对协方差矩阵进行求逆操作，用于EWA（Elliptical Weighted Average）算法
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
    // 计算2D协方差矩阵的特征值，用于计算屏幕空间的范围，以确定与之相交的瓦片
	float mid = 0.5f * (cov.x + cov.z); // 计算中间值
    // 计算特征值
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    // 计算长轴半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };	// 将相机中心从NDC平面拉回到图像平面
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr) {
        // （默认）如果未提供预计算的颜色，则球谐系数计算颜色
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

    // 存储计算的深度、半径、屏幕坐标等结果，用于下一步继续处理
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
    // 前三个将被用于计算高斯的指数部分从而得到 prob（查询点到该高斯的距离->prob，例如，若查询点位于该高斯的中心则 prob 为 1）。最后一个是该高斯本身的密度。
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


//! 光栅化最终的渲染步骤，渲染计算每个像素的颜色。
// 在一个block上协作，每个线程负责一个像素，在获取数据 与 光栅化数据之间交替。在这个过程中，每个像素的颜色是通过考虑所有影响该像素的高斯来计算的。
// (1) 通过计算当前线程所属的 tile 的范围，确定当前线程要处理的像素区域
// (2) 判断当前线程是否在有效像素范围内，如果不在，则将 done 设置为 true，表示该线程不执行渲染操作
// (3) 使用 __syncthreads_count 函数，统计当前块内 done 变量为 true 的线程数，如果全部线程都完成，跳出循环
// (4) 在每个迭代中，从全局内存中收集每个线程块对应的范围内的数据，包括点的索引、2D 坐标和锥体参数透明度。
// (5) 对当前线程块内的每个点，进行基于锥体参数的渲染，计算贡献并更新颜色。
// (6) 所有线程处理完毕后，将渲染结果写入 final_T、n_contrib 和 out_color
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)    // CUDA 启动核函数时使用的线程格和线程块的数量
renderCUDA(
	const uint2* __restrict__ ranges,   // 包含每个范围的起始和结束索引的数组
	const uint32_t* __restrict__ point_list,    // 包含了点的索引的数组f
	int W, int H,
	const float2* __restrict__ points_xy_image, // 包含每个点在屏幕上的坐标的数组
	const float* __restrict__ features,         // 包含每个点的颜色信息的数组
	const float4* __restrict__ conic_opacity,   // 包含每个点的锥体参数和透明度信息的数组
	float* __restrict__ final_T,                // 用于存储每个像素的最终颜色的数组。（多个叠加？）
	uint32_t* __restrict__ n_contrib,           // 用于存储每个像素的贡献计数的数组
	const float* __restrict__ bg_color,         // 如果提供了背景颜色，将其作为背景
	float* __restrict__ out_color)              //存储最终渲染结果的数组
{
	// Identify current tile and associated min/max pixel range.
    // 1.确定当前像素范围：
    // 这部分代码用于确定当前线程块要处理的像素范围，包括 pix_min 和 pix_max，并计算当前线程对应的像素坐标 pix
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
    // 2.判断当前线程是否在有效像素范围内：
    // 根据像素坐标判断当前线程是否在有效的图像范围内，如果不在，则将 done 设置为 true，表示该线程无需执行渲染操作
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
    // 3.加载点云数据处理范围：
    // 这部分代码加载当前线程块要处理的点云数据的范围，即 ranges 数组中对应的范围，并计算点云数据的迭代批次 rounds 和总共要处理的点数 toDo
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
    // 4. 初始化共享内存，分别定义三个共享内存数组，用于在每个线程块内共享数据
    // 每个线程取一个，并行读数据到 shared memory。然后每个线程都访问该shared memory，读取顺序一致。
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
    // 5.初始化渲染相关变量：
    // 初始化渲染所需的一些变量，包括当前像素颜色 C、贡献者数量
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
    // 6.迭代处理点云数据：
    // 在每个迭代中，处理一批点云数据。内部循环迭代每个点，进行基于锥体参数的渲染计算，并更新颜色信息
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // 代码使用 rounds 控制循环的迭代次数，每次迭代处理一批点云数据
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
        // 共享内存中获取点云数据：
        // 每个线程通过索引 progress 计算要加载的点云数据的索引 coll_id，然后从全局内存中加载到共享内存 collected_id、collected_xy 和 collected_conic_opacity 中。block.sync() 确保所有线程都加载完成
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
        // 迭代处理当前批次的点云数据
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            //在当前批次的循环中，每个线程处理一条点云数据
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
            // 计算当前点的投影坐标与锥体参数的差值：
            // 计算当前点在屏幕上的坐标 xy 与当前像素坐标 pixf 的差值，并使用锥体参数计算 power。
            // Resample using conic matrix (cf. "Surface
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
            // 计算论文中公式2的 alpha
			float alpha = min(0.99f, con_o.w * exp(power)); // opacity * 像素点出现在这个高斯的几率
			if (alpha < 1.0f / 255.0f)  // 太小了就当成透明的
				continue;
			float test_T = T * (1 - alpha); // alpha合成的系数
            // 累乘不透明度到一定的值，标记这个像素的渲染结束
			if (test_T < 0.0001f) {
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
            // 使用3D高斯进行渲染计算：更新颜色信息 C
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside) {
        //7. 写入最终渲染结果：
        // 如果当前线程在有效像素范围内，则将最终的渲染结果写入相应的缓冲区，包括 final_T、n_contrib 和 out_color
		final_T[pix_id] = T;    // 用于反向传播计算梯度
		n_contrib[pix_id] = last_contributor;   // 记录数量，用于提前停止计算
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// 渲染的主函数
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
    // 开始进入CUDA并行计算，将image分为多个block，每个block分配一个进程；每个block里面的pixel分配一个线程；
    // 对于每个block只排序一次，认为block里面的pixel都被block中的所有gaussian影响且顺序一样。
    // 在forward中，沿camera从前往后遍历gaussian，计算颜色累计值和透明度累计值，直到透明度累计超过1或者遍历完成，然后用背景色和颜色累计值和透明度累计值计算这个pixel的最终颜色。
    // 在backward中，遍历顺序与forward相反，从（之前记录下来的）最终透明度累计值和其对应的最后一个gaussian开始，从后往前算梯度。
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

// 预处理
void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
    // 调用CUDA核函数 preprocessCUDA为每个高斯进行预处理，为后续的光栅化做好准备
    // CUDA核函数的执行由函数参数确定。在CUDA核函数中，每个线程块由多个线程组成，负责处理其中的一部分数据，从而加速高斯光栅化的计算
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}