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

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
/**
 * 根据一个3D高斯的球谐系数，计算 当前相机中心 看向 该高斯中心 方向的RGB颜色值，(3,)
 * @param idx   当前高斯的索引
 * @param deg   当前的球谐阶数
 * @param max_coeffs    每个高斯的球谐系数个数=16
 * @param means 所有高斯 中心的世界坐标
 * @param campos 当前相机中心的世界坐标
 * @param shs    所有高斯的 球谐系数
 * @param clamped   geomState中记录高斯是否被裁剪的标志，即某位置为 True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
 */
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// 该函数基于zhang等人的论文"Differentiable Point-Based Radiance Fields for Efficient View Synthesis"中的代码实现
	glm::vec3 pos = means[idx];		// 当前高斯中心 的世界坐标
	glm::vec3 dir = pos - campos;	// 从 相机中心 指向 当前高斯中心的 单位向量
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;   // 获取当前高斯的球谐系数(16, 3)

    // 基函数(SH_C0、SH_C1等) * 系数(sh) = 最终的球谐函数

    // 计算当前高斯的0阶SH系数的颜色值，(3,)
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0) {
        // 当前的球谐阶数 > 0，则计算一阶SH系数的颜色值
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1) {
            // 当前的球谐阶数 > 1，则计算二阶SH系数的颜色值
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2) {
                // 当前的球谐阶数 > 2，则计算三阶SH系数的颜色值
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
    // 为结果颜色值加上一个偏移量，(3,)
	result += 0.5f;

    // 将RGB颜色值限制在正值范围内。如果计算的 当前相机看该高斯的RGB颜色值 < 0，则在 geomState的 clamped中记录其RGB对应的值为True
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);  // 返回值>0的RGB值
}

// Forward version of 2D covariance matrix computation
/**
 * 3D协方差矩阵 ==> 2D协方差矩阵:
 * 1. 世界坐标系到相机坐标：viewmatirx
 * 2. 视锥到立方体：雅克比矩阵
 * @param mean      该高斯中心 的世界坐标
 * @param focal_x   相机在焦x轴方向上的焦距，也是视锥体近平面的深度
 * @param focal_y
 * @param tan_fovx  tan(Fovx / 2)
 * @param tan_fovy
 * @param cov3D     该高斯的 世界坐标系下的 3D协方差矩阵
 * @param viewmatrix 观测变换矩阵，W2C
 * @return 像素坐标系下的协方差矩阵，维度为(2,2)，但只返回了上半角元素(3个)
 */
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29 and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// 雅克比矩阵在一个点附近才满足使用线性变换 近似 非线性变换的条件，而高斯的中心位置就是这个点，所以先求得3D高斯在相机坐标系下的位置）
	// 计算高斯中心在相机坐标系中的位置（在视锥中的位置）
	float3 t = transformPoint4x3(mean, viewmatrix);

    // 定义x和y方向的视锥限制
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
 * 计算该高斯的 3D协方差矩阵（从旋转和缩放计算，需注意旋转四元数的归一化）
 * @param scale 该高斯的 缩放因子
 * @param mod   缩放因子调整系数
 * @param rot   该高斯的 旋转四元数
 * @param cov3D 输出的 协方差矩阵
 */
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// 创建 缩放矩阵(3x3)
	glm::mat3 S = glm::mat3(1.0f);  // 初始化为一个3维的单位阵
	S[0][0] = mod * scale.x;    // 将缩放因子填入主对角线元素中
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// 将输入的四元数归一化 以正确表示 旋转（假设已经是单位四元数，因此不再进行额外的标准化）
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

    // 计算M矩阵，即缩放后的旋转矩阵
	glm::mat3 M = S * R;

	// 计算世界坐标系下的 协方差矩阵(3x3)：R^T S^T S R
	glm::mat3 Sigma = glm::transpose(M) * M;

	// 因为协方差矩阵是对阵矩阵，因此只需存储上半角元素
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
    int P,  // 所有高斯的个数
    int D,  // 当前的球谐阶数
    int M,  // 每个高斯的球谐系数个数=16
	const float* orig_points,   // 所有高斯 中心的世界坐标 数组，(x0, y0, z0, ..., xn, yn, zn)
	const glm::vec3* scales,    // 所有高斯的 缩放因子
	const float scale_modifier, // 缩放因子的调整系数
	const glm::vec4* rotations, // 所有高斯的 旋转四元数
	const float* opacities,     // 所有高斯的 不透明度
	const float* shs,           // 所有高斯的 球谐系数
	bool* clamped,              // geomState中记录高斯是否被裁剪的标志，即某位置为 True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
	const float* cov3D_precomp, // 因预计算的3D协方差矩阵默认是空tensor，则传入的是一个 NULL指针
	const float* colors_precomp,    // 因预计算的颜色默认是空tensor，则传入的是一个 NULL指针
	const float* viewmatrix,    // 观测变换矩阵，W2C
	const float* projmatrix,    // 观测变换矩阵 * 投影变换矩阵，W2NDC = W2C * C2NDC
	const glm::vec3* cam_pos,   // 当前相机中心的世界坐标
	const int W, int H,         // 输出图像的宽、高
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,                 // 输出的 所有高斯 投影在当前相机图像平面的最大半径 数组
	float2* points_xy_image,    // 输出的 所有高斯 中心在当前相机图像平面的二维坐标 数组
	float* depths,              // 输出的 所有高斯 中心在当前相机坐标系下的z值 数组
	float* cov3Ds,              // 输出的 所有高斯 在世界坐标系下的3D协方差矩阵 数组
	float* rgb,                 // 输出的 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
	float4* conic_opacity,      // 输出的 所有高斯 2D协方差的逆 和 不透明度 数组
	const dim3 grid,            // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
	uint32_t* tiles_touched,    // 输出的 所有高斯 在当前相机图像平面覆盖的线程块 tile的个数 数组
	bool prefiltered)           // 预滤除的标志，默认为False
{
    // 1. 获取当前线程在CUDA grid中的全局索引，即当前线程处理的高斯的索引
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)   // 一个线程只处理一个高斯，避免越界访问
		return;

    // 2. 初始化该高斯 在图像平面的最大投影半径、覆盖的tile数量为 0。如果这些值保持为0，说明该高斯不会影响最终渲染，不需要进一步处理
	radii[idx] = 0;
	tiles_touched[idx] = 0;

    // 3. 检查该高斯是否在当前相机的视锥体内
	float3 p_view;  // 计算的 该高斯中心在相机坐标系下的三维坐标
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        // 如果该高斯不在视锥体内，则直接返回
		return;

    // 4. 将该高斯投影到2D图像平面上
	// 4.1 将高斯的 中心 从3D变换到2D：包含观测变换、投影变换、视口变换、光栅化
    // (1) 该高斯中心的世界坐标
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// (2) 变换该中心坐标：世界坐标系 ==> NDC坐标系（4维齐次坐标）
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);   // projmatrix = 观测变换 * 投影变换
	float p_w = 1.0f / (p_hom.w + 0.0000001f);  // 齐次坐标的归一化因子
    // (3) 映射到范围为[-1,1]的正方体中的三维坐标，用于后续2D投影
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // 4.2 将该高斯的 协方差矩阵 从3D变换到2D：包含观测变换(viewmatrix)、投影变换中的视锥到立方体的变换(雅可比矩阵)
	// (1) 获取世界坐标系下的 3D协方差矩阵
	const float* cov3D;
	if (cov3D_precomp != nullptr) {
		// 如果提供了预计算的3D协方差矩阵，则直接使用它
		cov3D = cov3D_precomp + idx * 6;
	} else {
		// 默认未提供，则从缩放因子和旋转四元数中计算世界坐标系下的 3D协方差矩阵，并存储在conv3Ds数组中
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}
    // (2) 变换该3D协方差矩阵：世界坐标系 ==观测变换==> 相机坐标系 ==投影变换中视锥到立方体==> 立方体中
    // 2D协方差矩阵（只存了上半角元素，3个）
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 5. 计算2D协方差矩阵的 逆，用于EWA滤波算法
	float det = (cov.x * cov.z - cov.y * cov.y);    // 矩阵的行列式（sigma_x * sigma_y - sigma_xy^2）
	if (det == 0.0f)
        // 行列式为0，该矩阵 不可逆，则直接返回
		return;
	float det_inv = 1.f / det;
    // 2D协方差矩阵的逆矩阵（也只存了上半角元素，2x2的矩阵的取逆是 主对角线取反，次对角线取负，再除以行列式）
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };  // 取逆

    // 6. 计算该3D高斯投影在屏幕平面上的、扩展后的 投影圆所在的矩形框边界，最后再转换为 线程块的坐标。如果矩形覆盖0个tile，则退出
    // (1) 计算该高斯 投影在图像平面的最大半径
	float mid = 0.5f * (cov.x + cov.z); // 2D协方差矩阵主对角线元素的均值
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det)); // 2D协方差矩阵的特征值，即代表2D椭圆的长轴和短轴
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));  // 投影在图像平面的最大半径

    // (2) 计算该高斯 中心在图像平面的二维坐标（从 NDC平面 拉回到 图像平面）
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    // (3) 计算该高斯投影最大半径画的圆 在图像平面的影响范围（左上角和右下角坐标）对应在CUDA线程块（投影矩形）的边界
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);

    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        // 投影矩形面积=0，说明该高斯不会影响任何屏幕像素，则直接返回
		return;

    // 7. 如果提供了预计算的颜色，则直接使用
	if (colors_precomp == nullptr) {
        // 默认，未预计算颜色，则根据 该高斯的球谐系数 与 当前相机看该高斯的方向 计算该观测下的RGB颜色值，(3,)，同时如果某个RGB值<0，则在 clamped数组对应位置中置为 True
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

    // 8. 存储计算的深度、半径、屏幕坐标等结果，用于下一步继续处理
	depths[idx] = p_view.z; // 该高斯中心在相机坐标系下的z值
	radii[idx] = my_radius; // 该高斯投影在图像平面的最大半径
	points_xy_image[idx] = point_image; // 该高斯中心在图像平面的二维坐标
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };     // 该高斯的2D协方差的逆、不透明度
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // 该高斯投影最大半径画的圆 在屏幕空间覆盖的tile数量，用于渲染的优化
}


//! 渲染：计算每个 像素 的颜色（一个线程）
// 在一个block上协作，每个线程负责一个像素，在获取数据 与 光栅化数据之间交替。在这个过程中，每个像素的颜色是通过考虑所有影响该像素的高斯来计算的。
// (1) 通过计算当前线程所属的 tile 的范围，确定当前线程要处理的像素区域
// (2) 判断当前线程是否在有效像素范围内，如果不在，则将 done 设置为 true，表示该线程不执行渲染操作
// (3) 使用 __syncthreads_count 函数，统计当前块内 done 变量为 true 的线程数，如果全部线程都完成，跳出循环
// (4) 在每个迭代中，从全局内存中收集每个线程块对应的范围内的数据，包括点的索引、2D 坐标和锥体参数透明度。
// (5) 对当前线程块内的每个点，进行基于锥体参数的渲染，计算贡献并更新颜色。
// (6) 所有线程处理完毕后，将渲染结果写入 final_T、n_contrib 和 out_color
template <uint32_t CHANNELS>    // CHANNELS取3，即RGB三个通道
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)    // CUDA 启动核函数时使用的线程格和线程块的数量
renderCUDA(
	const uint2* __restrict__ ranges,   // 每个线程块 tile在排序后的高斯ID列表中的范围 数组，包含起始和结束索引
	const uint32_t* __restrict__ point_list,    // 按 tile、深度排序后的高斯的ID列表 的数组
	int W, int H,
	const float2* __restrict__ points_xy_image, // 所有高斯 中心在当前相机图像平面的二维坐标 的数组
	const float* __restrict__ features,         // 所有高斯 RGB颜色信息 的数组
	const float4* __restrict__ conic_opacity,   // 所有高斯 2D协方差矩阵的逆 和 不透明度 的数组
	float* __restrict__ final_T,                // 输出的 渲染过程后每个像素 pixel的最终透明度或透射率值 的数组
	uint32_t* __restrict__ n_contrib,           // 输出的 每个像素 pixel的最后一个贡献的高斯 的数组（？多少个高斯对该像素有贡献）
	const float* __restrict__ bg_color,         // 提供的背景颜色，默认为[1,1,1]，黑色
	float* __restrict__ out_color)              // 输出的 RGB图像
{
	// Identify current tile and associated min/max pixel range.
    // 1. 确定当前像素范围：
    // 这部分代码用于确定当前线程块要处理的像素范围，包括 pix_min 和 pix_max，并计算当前线程对应的像素坐标 pix
	auto block = cg::this_thread_block();   // 获取当前线程所处的 block（对应一个 tile）
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // 在水平方向上有多少个 block

    // block.group_index()：当前线程所处的 block在 grid中的三维索引。block.thread_index()：当前线程在 block中的三维索引
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };   // 当前处理的 tile的 左上角 像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };  // 当前处理的 tile的 右下角 像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前处理的 像素 在图像平面的坐标

    uint32_t pix_id = W * pix.y + pix.x;        // 当前处理的 像素 的索引
	float2 pixf = { (float)pix.x, (float)pix.y };   // 当前处理的 像素 在图像平面的坐标

    // 2. 判断当前线程是否在有效像素范围内：
    // 根据像素坐标判断当前线程是否在有效的图像范围内，如果不在，则将 done 设置为 true，表示该线程无需执行渲染操作
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
    // 3. 加载点云数据处理范围：
    // 这部分代码加载当前线程块要处理的点云数据的范围，即 ranges 数组中对应的范围，并计算点云数据的迭代批次 rounds 和总共要处理的点数 toDo
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];    // 当前处理的tile对应的3D gaussian的起始id和结束id
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;   // 还有多少3D gaussian需要处理

	// Allocate storage for batches of collectively fetched data.
    // 4. 初始化共享内存，分别定义三个共享内存数组，用于在每个线程块内共享数据
    // 每个线程取一个，并行读数据到 shared memory。然后每个线程都访问该shared memory，读取顺序一致。
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
    // 5. 初始化渲染相关变量：
    // 初始化渲染所需的一些变量，包括当前像素颜色 C、贡献者数量
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
    // 6. 迭代处理点云数据：
    // 在每个迭代中，处理一批点云数据。内部循环迭代每个点，进行基于锥体参数的渲染计算，并更新颜色信息
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // 代码使用 rounds 控制循环的迭代次数，每次迭代处理一批点云数据
		// End if entire block votes that it is done rasterizing
        // 检查是否所有线程块都已经完成渲染：
        // 通过 __syncthreads_count 统计已经完成渲染的线程数，如果整个线程块都已完成，则跳出循环
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
        // 共享内存中获取点云数据：
        // 每个线程通过索引 progress 计算要加载的点云数据的索引 coll_id，然后从全局内存中加载到共享内存 collected_id、collected_xy 和 collected_conic_opacity 中。block.sync() 确保所有线程都加载完成
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
            // 当前处理的3D gaussian的id
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
			float2 xy = collected_xy[j];    // 当前处理的2D gaussian在图像上的中心点坐标
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };    // 当前处理的2D gaussian的中心点到当前处理的pixel的offset
			float4 con_o = collected_conic_opacity[j];              // 当前处理的2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
            // 计算高斯分布的强度（或权重），用于确定像素在光栅化过程中的贡献程度
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
        // 7. 写入最终渲染结果：
        // 如果当前线程在有效像素范围内，则将最终的渲染结果写入相应的缓冲区，包括 final_T、n_contrib 和 out_color
		final_T[pix_id] = T;    // 用于反向传播计算梯度（？渲染过程后每个像素的最终透明度或透射率值）
		n_contrib[pix_id] = last_contributor;   // 记录数量，用于提前停止计算（？最后一个贡献的2D gaussian是谁）
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

//! 渲染的主函数
void FORWARD::render(
	const dim3 grid,    // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
    dim3 block,         // 线程块 block在水平和垂直方向上的线程数
	const uint2* ranges,        // 每个线程块 tile在排序后的高斯ID列表中的范围
	const uint32_t* point_list, // 排序后的高斯的ID列表
	int W, int H,
	const float2* means2D,  // 已计算的 所有高斯 中心在当前相机图像平面的二维坐标
	const float* colors,    // 每个3D高斯对应的RGB颜色
	const float4* conic_opacity,    // 已计算的 所有高斯 2D协方差矩阵的逆 和 不透明度
	float* final_T,         // 渲染过程后每个像素 pixel的最终透明度或透射率值
	uint32_t* n_contrib,    // 每个像素 pixel的最后一个贡献的高斯是谁
	const float* bg_color,  // 背景颜色，默认为[1,1,1]，黑色
	float* out_color)       // 输出的 RGB图像
{
    // 开始进入CUDA并行计算，将图像分为多个线程块（分配一个 进程）；每个线程块为每个像素分配一个线程；
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

/**
 * 调用CUDA核函数 preprocessCUDA对每个高斯进行预处理和投影
 * (1) 计算高斯投影到图像平面的最大半径 radii
 * (2) 计算投影椭圆涉及到的tile的坐标和个数，以高效渲染
 */
void FORWARD::preprocess(
    int P,      // 所有高斯的个数
    int D,      // 当前的球谐阶数
    int M,      // 每个高斯的球谐系数个数=16
	const float* means3D,   // 所有高斯 中心的世界坐标 数组，(x0, y0, z0, ..., xn, yn, zn)
	const glm::vec3* scales,    // 所有高斯的 缩放因子
	const float scale_modifier, // 缩放因子的调整系数
	const glm::vec4* rotations, // 所有高斯的 旋转四元数
	const float* opacities,     // 所有高斯的 不透明度
	const float* shs,           // 所有高斯的 球谐系数
	bool* clamped,              // geomState中记录高斯是否被裁剪的标志，即某位置为 True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
	const float* cov3D_precomp, // 因预计算的3D协方差矩阵默认是空tensor，则传入的是一个 NULL指针
	const float* colors_precomp,    // 因预计算的颜色默认是空tensor，则传入的是一个 NULL指针
	const float* viewmatrix,    // 观测变换矩阵，W2C
	const float* projmatrix,    // 观测变换矩阵 * 投影变换矩阵，W2NDC = W2C * C2NDC
	const glm::vec3* cam_pos,   // 当前相机中心的世界坐标
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,   // tan(fov_x/2)和tan(fov_y/2)
	int* radii,             // 输出的 所有高斯 投影在当前相机图像平面的最大半径 数组
	float2* means2D,        // 输出的 所有高斯 中心在当前相机图像平面的二维坐标 数组
	float* depths,          // 输出的 所有高斯 中心在当前相机坐标系下的z值 数组
	float* cov3Ds,          // 输出的 所有高斯 在世界坐标系下的3D协方差矩阵 数组
	float* rgb,             // 输出的 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
	float4* conic_opacity,  // 输出的 所有高斯 2D协方差的逆 和 不透明度 数组
	const dim3 grid,        // CUDA网格的维度，grid.x是网格在x方向上的线程块数，grid.y是网格在y方向上的线程块数
	uint32_t* tiles_touched,    // 输出的 所有高斯 在当前相机图像平面覆盖的线程块 tile的个数 数组
	bool prefiltered)       // 预滤除的标志，默认为False
{
    /**
     * 核函数使用__global__修饰符声明。这表明该函数是一个核函数（只能在 GPU上执行，不能在 CPU上执行）
     * 调用核函数时，需要使用特殊语法 << <numBlocks, blockSize> >>(data) 来指定执行配置
     * 这个配置包括两个部分：
     *    numBlocks：指定了多少个块（block）组成网格（grid），整个网格代表了所有并行执行单元的集合；
     *    blockSize：每个块中有多少个线程。块内的线程可以共享数据并协作执行任务。
     *
     * 线程ID：在CUDA核函数中，每个线程都会被分配一个唯一的线程ID。这个ID用于区分同一个核函数中不同的执行线程，使得每个线程可以处理数据的不同部分。例如，在处理数组时，线程ID可以用来确定每个线程负责处理数组中的哪个元素。
     * 获取线程ID：线程ID可以通过核函数的内置变量threadIdx来获取。在一维配置中，threadIdx.x表示当前线程的ID。如果使用二维或三维的块配置，还可以使用threadIdx.y和threadIdx.z。
     */
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