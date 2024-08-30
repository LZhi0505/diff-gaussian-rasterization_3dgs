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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

/**
 * RasterizeGaussiansCUDA、RasterizeGaussiansBackwardCUDA 分别使用CudaRasterizer::Rasterizer类的 forward、backward函数。
 * Rasterizer类在cuda_rasterizer文件夹下的 rasterizer.h、rasterizer_impl.h、rasterizer_impl.cu中定义
 */

// 创建并返回一个 lambda 表达式，该表达式用于调整 torch::Tensor对象（内存缓冲区）的大小，并返回一个指向它数据的原始指针
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) { //输入的张量为 要动态管理的内存缓冲区
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});  // 调整t的大小为N
		return reinterpret_cast<char*>(t.contiguous().data_ptr());  // 返回一个指向张量t起始位置的char*指针
    };
    return lambda;
}

/**
 * 光栅化（前向传播）
 * @return: 一个元组，包含：渲染的高斯的个数 rendered、渲染的RGB图像 out_color、每个高斯在当前图像平面上的投影半径 radii、用于管理内存缓冲区的三个Tensor：geomBuffer、binningBuffer、imgBuffer
 */
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,    // 背景颜色，默认为[1,1,1]，黑色
	const torch::Tensor& means3D,   // 所有高斯 中心的世界坐标
    const torch::Tensor& colors,    // 预计算的颜色，默认是空tensor，后续在光栅化预处理阶段计算
    const torch::Tensor& opacity,   // 所有高斯的 不透明度
	const torch::Tensor& scales,    // 所有高斯的 缩放因子
	const torch::Tensor& rotations, // 所有高斯的 旋转四元数
	const float scale_modifier,     // 缩放因子调节系数
	const torch::Tensor& cov3D_precomp, // 预计算的3D协方差矩阵，默认为空tensor，后续在光栅化预处理阶段计算
	const torch::Tensor& viewmatrix,    // 观测变换矩阵，W2C
	const torch::Tensor& projmatrix,    // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,    // 所有高斯的 球谐系数，(N,16,3)
	const int degree,           // 当前的球谐阶数
	const torch::Tensor& campos,    // 当前相机中心的世界坐标
	const bool prefiltered,     // 预滤除的标志，默认为False
	const bool debug)           // 默认为False
{
  // 1. 检查所有高斯中心世界坐标 tensor的维度必须是(N,3)
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);    // 所有高斯的个数
  const int H = image_height;       // 图像高度
  const int W = image_width;        // 图像宽度

  // 2. 根据张量 means3D的数据类型，创建 int32 和 float32数据类型，分别用于整数和浮点数
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  // 3. 初始化输出Tensor：渲染的RGB图像 out_color (3,H,W) 和 每个高斯在当前图像平面上的投影半径 radii (N,) 为全 0 Tensor
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  // 4. 创建用于管理内存分配的辅助函数
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));         // 存储所有高斯的参数的 tensor：包括中心位置、缩放因子、旋转四元数、不透明度
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  // geomFunc、binningFunc、imgFunc 是三个函数指针，其指向的函数可以动态调整内存缓冲区的大小
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0; // 初始化渲染的高斯个数为0

  if(P != 0) {
      // 场景中存在高斯，则进行光栅化

      // 如果参数中输入了所有高斯的球谐系数，则 M=每个高斯的球谐系数个数=16；否则 M=0
	  int M = 0;
	  if(sh.size(0) != 0) {
		M = sh.size(1);
      }

      //! 5. 实际可微光栅化的前向渲染，返回：渲染的高斯的个数 rendered、渲染的RGB图像 out_color、每个高斯在当前图像平面上的投影半径 radii、用于管理内存缓冲区的三个Tensor：geomBuffer、binningBuffer、imgBuffer
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,   // 调整内存缓冲区的函数指针
		binningFunc,
		imgFunc,
	    P,          // 所有高斯的个数
        degree,     // 当前的球谐阶数
        M,          // 每个高斯的球谐系数个数=16
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),     // PyTorch Tensor 默认使用 row-major内存布局，而一些计算库如 CUDA更喜欢 column-major布局，通过 contiguous()可以确保数据在内存中是连续的
		sh.contiguous().data_ptr<float>(),      // CudaRasterizer::Rasterizer::forward需要C风格的原始指针作为输入，而不是Pytorch Tensor对象
		colors.contiguous().data<float>(),      // data<float>() 方法会返回 Tensor中数据的原始指针，同时将数据类型转换为 float*
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),   // cov3D_precomp默认是空Tensor，则传入一个 NULL指针
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),   // 输出的 颜色图像，(3,H,W)
		radii.contiguous().data<int>(),         // 输出的 在图像平面上的投影半径(N,)
		debug);
  }

  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}