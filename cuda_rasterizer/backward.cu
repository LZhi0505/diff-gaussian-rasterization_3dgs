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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


/**
 * 对每个高斯，将球谐函数转换为RGB的 反向传播
 * @param idx   当前高斯的索引
 * @param deg   指定的球谐函数的 阶数
 * @param max_coeffs    球谐函数的系数个数
 * @param means         每个高斯中心的 世界坐标
 * @param campos        当前相机的世界坐标
 * @param shs           每个高斯的 球谐系数
 * @param clamped       每个高斯是否需要进行截断
 * @param dL_dcolor     目标颜色对 RGB颜色空间 的导数
 * @param dL_dmeans     目标颜色对 3D高斯中心 的导数
 * @param dL_dshs       目标颜色对 球谐函数系数 的导数
 */
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
    // 计算 相机到3D高斯中心 的单位向量
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// 使用PyTorch rule进行截断：如果clamped为True，表示需要截断，则其梯度置为0
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;     // 对球鞋系数 0 阶的梯度
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


/**
 * 将 2D协方差矩阵 的梯度 传播到 高斯中心3D世界坐标、3D协方差矩阵 的梯度（由于计算量较大，在其他反传步骤之前作为单独的kernel进行计算）
 */
__global__ void computeCov2DCUDA(
    int P,      // 所有高斯的个数
	const float3* means,    // 所有高斯 中心的世界坐标
	const int* radii,       // 所有高斯 在图像平面上的投影半径
	const float* cov3Ds,    // 所有高斯 在世界坐标系下的3D协方差矩阵
	const float h_x, float h_y,     // fx, fy
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,   // 观测变换矩阵，W2C
	const float* dL_dconics,    // 反传渲染部分计算的 loss对所有高斯 2D协方差矩阵 的梯度
	float3* dL_dmeans,      // 输出的 loss对所有高斯 中心世界坐标 的梯度
	float* dL_dcov)         // 输出的 loss对所有高斯 3D协方差矩阵 的梯度
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


/**
 * 将 高斯中心2D投影像素坐标、高斯中心3D世界坐标、颜色、3D协方差矩阵 的梯度 传播到 球谐系数、轴长、旋转
 */
template<int C>
__global__ void preprocessCUDA(
	int P,      // 所有高斯的个数
    int D,      // 当前的球谐阶数
    int M,      // 每个高斯的球谐系数个数=16
	const float3* means,    // 所有高斯 中心的世界坐标
	const int* radii,       // 所有高斯 在图像平面上的投影半径
	const float* shs,       // 所有高斯的 球谐系数，(N,16,3)
	const bool* clamped,    // 所有高斯 是否被裁剪的标志 数组，某位置为True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
	const glm::vec3* scales,    // 所有高斯的 缩放因子
	const glm::vec4* rotations, // 所有高斯的 旋转四元数
	const float scale_modifier, // 缩放因子调节系数
	const float* proj,      // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	const glm::vec3* campos,    // 当前相机中心的世界坐标
	const float3* dL_dmean2D,   // 输入的 反传渲染部分计算的 loss对所有高斯 中心2D投影像素坐标 的梯度
	glm::vec3* dL_dmeans,       // 输入的 前一步计算的 loss对所有高斯 中心世界坐标 的梯度
	float* dL_dcolor,           // 输入的 反传渲染部分计算的 loss对所有高斯 RGB颜色 的梯度
	float* dL_dcov3D,           // 输入的 前一步计算的 loss对所有高斯 3D协方差矩阵 的梯度
	float* dL_dsh,          // 输出的 loss对所有高斯 球谐系数 的梯度
	glm::vec3* dL_dscale,   // 输出的 loss对所有高斯 缩放因子 的梯度
	glm::vec4* dL_drot)     // 输出的 loss对所有高斯 旋转四元数 的梯度
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

    // 这里的xyz得先变换到相机系哈,再按照上面的公式计算;

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}


/**
 * 反向传播中的 渲染
 * @tparam C = 3，即RGB三个通道
 */
template <uint32_t C, uint32_t MAP_N>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,   // 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile_ID；值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数。也可以用[x,y)在排序后的values列表中索引到该tile触及的所有高斯ID
	const uint32_t* __restrict__ point_list,    // 排序后的 values列表，每个元素是按（大顺序：各tile_ID，小顺序：落在该tile内各高斯的深度）排序后的 高斯ID
	int W, int H,
	const float* __restrict__ bg_color,     // 背景颜色，默认为[0,0,0]，黑色
	const float2* __restrict__ points_xy_image,     // 所有高斯 中心在当前相机图像平面的二维坐标 数组
	const float4* __restrict__ conic_opacity,       // 所有高斯 2D协方差的逆 和 不透明度 数组
	const float* __restrict__ colors,       // 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
    const float* __restrict__ all_maps,     // 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
    const float* __restrict__ all_map_pixels,   // forward输出的 5通道tensor，[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
	const float* __restrict__ final_Ts,     // 渲染后每个像素 pixel的 累积的透射率 的数组
	const uint32_t* __restrict__ n_contrib,     // 渲染每个像素 pixel穿过的高斯的个数，也是最后一个对渲染该像素RGB值 有贡献的高斯ID 的数组
	const float* __restrict__ dL_dpixels,   // 输入的 loss对渲染的RGB图像中每个像素颜色 的梯度（优化器输出的值，由优化器在训练迭代中自动计算）
    const float* __restrict__ dL_dout_all_maps,     // 输入的 loss对forward输出的 5通道tensor（[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离）的梯度
    const float* __restrict__ dL_dout_plane_depths, // 输入的 loss对渲染的无偏深度图 的梯度
	float3* __restrict__ dL_dmean2D,    // 输出的 loss对所有高斯 中心2D投影像素坐标 的梯度
    float3* __restrict__ dL_dmean2D_abs,    // 输出的 loss对所有高斯 中心2D投影像素坐标 的梯度 的绝对值
	float4* __restrict__ dL_dconic2D,   // 输出的 loss对所有高斯 2D协方差矩阵 的梯度
	float* __restrict__ dL_dopacity,    // 输出的 loss对所有高斯 不透明度 的梯度
	float* __restrict__ dL_dcolors,     // 输出的 loss对所有高斯 颜色 的梯度
    float* __restrict__ dL_dall_map,    // 输出的 loss对所有高斯 5通道tensor（法向量、贡献度、光心到高斯法切平面距离）的梯度
    const bool render_geo)      // 是否要渲染 深度图和法向量图的标志，默认为False
{
    // 重新进行光栅化计算，计算所需的块信息
    // 1. 确定当前block处理的 tile的像素范围
	auto block = cg::this_thread_block();   // 当前线程所处的 block（对应一个 tile）

    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;     // 在水平方向上有多少个 block

    // 当前处理的tile的 左上角像素的坐标、右下角像素的坐标
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    // 当前处理的 像素 在像素平面的坐标
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };

    const uint32_t pix_id = W * pix.y + pix.x;  // 当前处理的 像素 在像素平面的　索引
	const float2 pixf = { (float)pix.x, (float)pix.y }; // 当前处理的 像素 在像素平面的坐标(float)

    const float2 ray = { (pixf.x - W * 0.5) / fx, (pixf.y - H * 0.5) / fy };    // 当前处理的 像素 在Z=1平面的 坐标

    // 2. 判断当前处理的 像素 是否在图像有效像素范围内
	const bool inside = pix.x < W && pix.y < H;

    // 3. 计算当前tile触及的高斯个数，太多，则分rounds批渲染
    // 根据当前处理的 tile_ID，获取该tile在排序后的keys列表中的起始、终止位置，[x,y)。个数y-x：投影到该tile上的高斯的个数。
    // 也可以用[x,y)在排序后的values列表中索引到该tile触及的所有高斯ID
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // 高斯个数过多，则分批处理，每批最多处理 BLOCK_SIZE=16*16个高斯

	bool done = !inside;
	int toDo = range.y - range.x;   // 当前tile还未处理的 高斯的个数 = 落在该tile上的高斯的个数

    // 4. 初始化同一block中的各线程共享的四个显存数组，用于在该block内共享数据
	__shared__ int collected_id[BLOCK_SIZE];    // 记录各线程处理的 高斯ID
	__shared__ float2 collected_xy[BLOCK_SIZE]; // 记录各线程处理的 高斯中心在当前相机图像平面的 像素坐标
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];  // 记录各线程处理的 高斯的 2D协方差矩阵的 逆 和 不透明度
	__shared__ float collected_colors[C * BLOCK_SIZE];      // 记录各线程处理的 高斯 在当前相机中心的观测方向下 的RGB三个值
    __shared__ float collected_all_maps[MAP_N * BLOCK_SIZE];

    //! 核心：根据α-blending的公式，手推每个变量的反向传播公式，推导过程可参考论文里的附录

    // 从前向传播中取出 渲染当前像素后的 累积透射率 = 该像素射线上所有高斯的(1 - α)的乘积
	const float T_final = inside ? final_Ts[pix_id] : 0;

    float T = T_final;  // 经当前高斯 之前的 光线剩余能量（累积的透射率）

    // 从最后面的高斯开始
	uint32_t contributor = toDo;    // 当前跟踪的高斯ID = 该tile触及的最后一个的高斯ID = 该tile触及的高斯个数
    const int last_contributor = inside ? n_contrib[pix_id] : 0;    // 从前向传播中取出 该像素穿过的高斯的个数 = 最后一个对渲染该像素RGB值 有贡献的高斯ID

	float accum_rec[C] = { 0 };     // 当前高斯之后所有高斯的 渲染颜色
    float accum_all_map[MAP_N] = { 0 }; // 当前高斯之后所有高斯的 5通道tensor

    float dL_dpixel[C];             // loss对当前像素 RGB三个值 的梯度
    float dL_dout_all_map[MAP_N];   // loss对当前像素 5通道tensor（法向量、对渲染有贡献的所有高斯累加的贡献度、相机光心到当前像素穿过的所有高斯法向量垂直平面的 距离）的梯度
    // float grad_sum = 0;

	if (inside)
    {
        // 取出 loss对当前像素 RGB三个值的 梯度
        for (int i = 0; i < C; i++)
        {
            dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
            // grad_sum += fabs(dL_dpixel[i]);
        }

        if(render_geo)
        {
            // loss对forward输出的 5通道tensor（[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离）的梯度
            for (int i = 0; i < MAP_N; i++)
            {
                dL_dout_all_map[i] = dL_dout_all_maps[i * H * W + pix_id];
                // grad_sum += fabs(dL_dout_all_map[i]);
            }

            const float3 normal = {all_map_pixels[pix_id], all_map_pixels[H * W + pix_id], all_map_pixels[2 * H * W + pix_id]};     // 当前像素的 法向量
            const float distance = all_map_pixels[4 * H * W + pix_id];  // 相机光心 到 当前像素穿过的所有高斯法向量垂直平面的 距离
            const float tmp = (normal.x * ray.x + normal.y * ray.y + normal.z + 1.0e-8);    // cosθ = 当前像素在Z=1平面的坐标 投影到 法向量上的长度
            // loss对 相机光心到当前像素穿过的所有高斯法向量垂直平面距离 的梯度
            dL_dout_all_map[MAP_N-1] += (-dL_dout_plane_depths[pix_id] / tmp);
            // loss对 当前像素法向量 的梯度
            dL_dout_all_map[0] += dL_dout_plane_depths[pix_id] * (distance / (tmp * tmp) * ray.x);
            dL_dout_all_map[1] += dL_dout_plane_depths[pix_id] * (distance / (tmp * tmp) * ray.y);
            dL_dout_all_map[2] += dL_dout_plane_depths[pix_id] * (distance / (tmp * tmp));
        }
    }

	float last_alpha = 0;           // 迭代记录的 后一个高斯的 不透明度
	float last_color[C] = { 0 };    // 迭代记录的 后一个高斯的 颜色
    float last_all_map[MAP_N] = { 0 };  // 迭代记录的 后一个高斯的 5通道tensor（法向量、对渲染有贡献的所有高斯累加的贡献度、相机光心到当前像素穿过的所有高斯法向量垂直平面的 距离）

    //! 从后面开始加载辅助数据到共享内存中，并以相反顺序加载

    // 像素坐标的梯度。归一化屏幕空间视窗坐标(-1, 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

    // 遍历当前tile触及的高斯
    // 外循环：迭代分批渲染任务，每批最多处理 BLOCK_SIZE = 16*16个高斯
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
        // 将辅助数据加载到共享内存中，从后面开始并以相反的顺序加载它们。
		block.sync();   // 同步当前block下的所有线程
		const int progress = i * BLOCK_SIZE + block.thread_rank();  // 当前线程处理的高斯的全局ID。block.thread_rank()：当前线程在该 block内的ID，区间为[0, BLOCK_SIZE)
        // 当前线程处理的高斯不越界
        if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1]; // 以逆序方式获取高斯ID（从后往前）

            collected_id[block.thread_rank()] = coll_id;    // 当前线程处理的 高斯ID
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];   // 当前线程处理的 高斯中心在当前相机图像平面的 像素坐标
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];  // 当前线程处理的 高斯 2D协方差的逆 和 不透明度

            for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];   // 当前线程处理的 高斯 在当前相机中心的观测方向下 的RGB三个值

            if (render_geo)
            {
                // 当前线程处理的 高斯 [0-2]: 法向量（相机坐标系）、[3]: 全1.0、[4]: 相机光心 到 当前高斯法向量垂直平面的 距离
                for (int i = 0; i < MAP_N; i++)
                    collected_all_maps[i * BLOCK_SIZE + block.thread_rank()] = all_maps[coll_id * MAP_N + i];
            }
		}
		block.sync();

        // 内循环：每个线程遍历 当前批次的高斯，进行基于锥体参数的渲染计算，并更新颜色信息
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
            // 当前跟踪的 高斯ID
			contributor--;
            // 如果当前高斯位于 当前处理的像素的最后一个贡献高斯之后，就跳过它
			if (contributor >= last_contributor)
				continue;

            // 计算α-blending值（如forward）
			const float2 xy = collected_xy[j];  // 当前高斯中心在图像平面的 像素坐标
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };  // 当前处理的像素 到 当前2D高斯中心像素坐标的 位移向量
			const float4 con_o = collected_conic_opacity[j];    // 当前高斯的 2D协方差矩阵的逆 和 不透明度，x、y、z: 分别是2D协方差逆矩阵的上半对角元素, w：不透明度

            // 当前2D高斯分布的指数部分：-1/2 d^T Σ^-1 d
            const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
            // 当前高斯的 alpha = 高斯的不透明度(包含2D高斯投影分布前面的常数部分) * 2D高斯的投影分布
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);  // 更新 光线到当前高斯剩余的能量
			const float dchannel_dcolor = alpha * T;    // 像素颜色对 当前高斯颜色 的梯度

            // 将梯度传播到每个高斯的颜色，并保留相对于 alpha（高斯和像素对的blending因子）的梯度。
            // 计算loss对高斯的 alpha、颜色 的梯度
			float dL_dalpha = 0.0f;     // loss对当前高斯alpha的 梯度
			const int global_id = collected_id[j];  // 当前高斯ID

            // 遍历当前高斯的RGB三个通道，分别计算
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];  // 当前高斯某个通道的 颜色值

                // 更新 当前高斯之后所有高斯的 渲染颜色（用于下一次迭代）
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];   // = 后一个高斯的不透明度 * 后一个高斯的颜色 + (1 - 后一个高斯的不透明度) * 后一个高斯之后所有高斯的 颜色

                last_color[ch] = c;     // 更新 后一个高斯的 颜色（用于下一次迭代）

				const float dL_dchannel = dL_dpixel[ch];    // loss对当前像素某个通道颜色的 梯度

                // (1) 累加 RGB三个通道的loss 对当前高斯 最终不透明度 的梯度
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

                // (2) 更新 loss对当前高斯 颜色 的梯度
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);    // atomicAdd：CUDA编程中的一种原子 加 操作，确保在并发操作时多个线程访问同一内存地址的，执行安全、准确的 加法运算。因为这个像素可能只是受此高斯影响的众多像素之一
			}

            if (render_geo)
            {
                for (int ch = 0; ch < MAP_N; ch++)
                {
                    // 循环获取 当前线程处理的高斯的 5通道tensor（法向量相机坐标系、全1.0、光心到高斯法切平面的距离）
                    const float c = collected_all_maps[ch * BLOCK_SIZE + j];
                    // 更新 当前高斯之后所有高斯的 5通道tensor（用于下一次迭代）
                    accum_all_map[ch] = last_alpha * last_all_map[ch] + (1.f - last_alpha) * accum_all_map[ch]; // = 后一个高斯的不透明度 * 后一个高斯的5通道tensor + (1 - 后一个高斯的不透明度) * 后一个高斯之后所有高斯的 5通道tensor
                    last_all_map[ch] = c;   // 更新 后一个高斯的 5通道tensor（用于下一次迭代）

                    const float dL_dchannel = dL_dout_all_map[ch];      // loss对 当前像素某个通道tensor 的梯度

                    // (1) 累加 5通道tensor的loss 对当前高斯 最终不透明度 的梯度
                    dL_dalpha += (c - accum_all_map[ch]) * dL_dchannel;

                    // (2) 更新 loss对当前高斯 5通道tensor（法向量、贡献度、光心到高斯法切平面距离）的梯度
                    atomicAdd(&(dL_dall_map[global_id * MAP_N + ch]), dchannel_dcolor * dL_dchannel);
                }
            }
			dL_dalpha *= T;

            // 更新 后一个高斯的 不透明度（用于下一次迭代）
			last_alpha = alpha;

            // 在混合完所有高斯的颜色后，不透明度alpha还会影响 添加的背景颜色，因此还需计算对alpha的梯度还需考虑背景部分
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];

			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


            // 有用的可重用临时变量
			const float dL_dG = con_o.w * dL_dalpha;    // loss对当前高斯 2D投影分布 的梯度
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

            // (3.1) 更新 loss对当前高斯中心 2D投影像素坐标 的梯度
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

            // (3.2) 更新 loss对当前高斯中心 2D投影像素坐标 的梯度 的绝对值
            atomicAdd(&dL_dmean2D_abs[global_id].x, fabs(dL_dG * dG_ddelx * ddelx_dx));
            atomicAdd(&dL_dmean2D_abs[global_id].y, fabs(dL_dG * dG_ddely * ddely_dy));

            // (4) 更新 loss对当前高斯的 2D协方差矩阵 的梯度
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

            // (5) 更新 loss对当前高斯 不透明度 的梯度
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P,      // 所有高斯的个数
    int D,      // 当前的球谐阶数
    int M,      // 每个高斯的球谐系数个数=16
	const float3* means3D,      // 所有高斯 中心的世界坐标
	const int* radii,           // 所有高斯 在图像平面上的投影半径
	const float* shs,           // 所有高斯的 球谐系数，(N,16,3)
	const bool* clamped,        // 所有高斯 是否被裁剪的标志 数组，某位置为True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
	const glm::vec3* scales,        // 所有高斯的 缩放因子
	const glm::vec4* rotations,     // 所有高斯的 旋转四元数
	const float scale_modifier,     // 缩放因子调节系数
	const float* cov3Ds,        // 所有高斯 在世界坐标系下的3D协方差矩阵
	const float* viewmatrix,    // 观测变换矩阵，W2C
	const float* projmatrix,    // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,    // 当前相机中心的世界坐标
	const float3* dL_dmean2D,   // 反传渲染部分计算的 loss对所有高斯 中心2D投影像素坐标 的梯度
	const float* dL_dconic,     // 反传渲染部分计算的 loss对所有高斯 2D协方差矩阵 的梯度
	glm::vec3* dL_dmean3D,  // 输出的 loss对所有高斯 中心世界坐标 的梯度
	float* dL_dcolor,           // 反传渲染部分计算的 loss对所有高斯 RGB颜色 的梯度
	float* dL_dcov3D,       // 输出的 loss对所有高斯 3D协方差矩阵 的梯度
	float* dL_dsh,          // 输出的 loss对所有高斯 球谐系数 的梯度
	glm::vec3* dL_dscale,   // 输出的 loss对所有高斯 缩放因子 的梯度
	glm::vec4* dL_drot)     // 输出的 loss对所有高斯 旋转四元数 的梯度
{
    // 将 2D协方差矩阵 的梯度 传播到 高斯中心3D世界坐标、3D协方差矩阵 的梯度（因为计算过程较长，所以单独计算）
    // 完成后，损失梯度相对于3D均值已被修改，且相对于3D协方差矩阵的梯度已被计算。
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

    // 将 高斯中心2D投影像素坐标、高斯中心3D世界坐标、颜色、3D协方差矩阵 的梯度 传播到 球谐系数、轴长、旋转
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid,    // 线程块 block（tile）的维度，(W/16, H/16, 1)
    const dim3 block,   // 一个block中 线程thread的维度，(16, 16, 1)
	const uint2* ranges,    // 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile_ID；值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数
	const uint32_t* point_list, // 按深度排序后的 所有高斯覆盖的tile的 values列表，每个元素是 对应高斯的ID
	int W, int H,
    float fx, float fy,
	const float* bg_color,  // 背景颜色，默认为[0,0,0]，黑色
	const float2* means2D,  // 所有高斯 中心投影在当前相机图像平面的二维坐标 数组
	const float4* conic_opacity,    // 所有高斯 2D协方差的逆 和 不透明度 数组
	const float* colors,    // 默认是 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
    const float* all_maps,      // 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
    const float* all_map_pixels,    // forward输出的 5通道tensor，[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
	const float* final_Ts,  // 渲染后每个像素 pixel的 累积的透射率 的数组
	const uint32_t* n_contrib,  // 渲染每个像素 pixel穿过的高斯的个数，也是最后一个对渲染该像素RGB值 有贡献的高斯ID 的数组
	const float* dL_dpixels,            // 输入的 loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
    const float* dL_dout_all_map,       // 输入的 loss对forward输出的 5通道tensor（[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离）的梯度
    const float* dL_dout_plane_depth,   // 输入的 loss对渲染的无偏深度图 的梯度
	float3* dL_dmean2D,     // 输出的 loss对所有高斯 中心2D投影像素坐标 的梯度
    float3* dL_dmean2D_abs, // 输出的 loss对所有高斯 中心2D投影像素坐标 的梯度 的绝对值
	float4* dL_dconic2D,    // 输出的 loss对所有高斯 2D协方差矩阵 的梯度
	float* dL_dopacity,     // 输出的 loss对所有高斯 不透明度 的梯度
	float* dL_dcolors,      // 输出的 loss对所有高斯 在当前相机中心的观测方向下 的RGB颜色值 的梯度
    float* dL_dall_map,     // 输出的 loss对所有高斯 5通道tensor（法向量、贡献度、光心到高斯法切平面距离）的梯度
    const bool render_geo)  // 是否要渲染 深度图和法向量图的标志，默认为False
{
    // 调用CUDA内核函数执行光栅化过程
	renderCUDA<NUM_CHANNELS, NUM_ALL_MAP> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
        fx, fy,
		bg_color,
		means2D,
		conic_opacity,
		colors,
        all_maps,
        all_map_pixels,
		final_Ts,
		n_contrib,
		dL_dpixels,
        dL_dout_all_map,
        dL_dout_plane_depth,
		dL_dmean2D,
        dL_dmean2D_abs,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
        dL_dall_map,
        render_geo
		);
}