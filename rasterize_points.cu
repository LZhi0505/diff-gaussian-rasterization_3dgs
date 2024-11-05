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
 * @return: 一个元组，包含：   rendered：渲染的tile的个数
 *                          out_color：渲染的RGB图像
 *                          radii：每个高斯投影在当前相机图像平面上的最大半径 数组
 *                          geomBuffer：所有高斯 几何数据的 tensor：包括2D中心像素坐标、相机坐标系下的深度、3D协方差矩阵等
 *                          binningBuffer：存储所有高斯 排序数据的 tensor：包括未排序和排序后的 所有高斯覆盖的tile的 keys、values列表
 *                          imgBuffer：存储所有高斯 渲染后数据的 tensor：包括累积的透射率、最后一个贡献的高斯ID
 */
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,    // 背景颜色，默认为[0,0,0]，黑色
	const torch::Tensor& means3D,   // 所有高斯 中心的世界坐标
    const torch::Tensor& colors,    // 预计算的颜色，默认是空tensor，后续在光栅化预处理阶段计算
    const torch::Tensor& opacity,   // 所有高斯的 不透明度
	const torch::Tensor& scales,    // 所有高斯的 缩放因子
	const torch::Tensor& rotations, // 所有高斯的 旋转四元数
	const float scale_modifier,     // 缩放因子调节系数
	const torch::Tensor& cov3D_precomp, // 预计算的3D协方差矩阵，默认为空tensor，后续在光栅化预处理阶段计算
    const torch::Tensor& all_map,   // 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
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
    const bool render_geo,      // 是否要渲染 深度图和法向量图的标志，默认为False
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

  // 3. 初始化 输出Tensor为 全0 Tensor
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);         // 输出的 渲染的RGB图像 out_color (3,H,W)
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));    // 输出的 每个高斯在当前图像平面上的投影半径 radii (N,)
  torch::Tensor out_observe = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));  // 输出的 所有高斯 渲染时在透射率>0.5之前 对某像素有贡献的 像素个数 数组，(N,)
  torch::Tensor out_all_map = torch::full({NUM_ALL_MAP, H, W}, 0, float_opts);      // 输出的 5通道tensor，[0-2]：渲染的法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
  torch::Tensor out_plane_depth = torch::full({1, H, W}, 0, float_opts);            // 输出的 无偏深度图（相机坐标系）

  // 4. 创建用于管理内存分配的辅助函数
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);

  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));         // 存储所有高斯 几何数据的 tensor：包括2D中心像素坐标、相机坐标系下的深度、3D协方差矩阵等
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));      // 存储所有高斯 排序数据的 tensor：包括未排序和排序后的 所有高斯覆盖的tile的 keys、values列表
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));          // 存储所有高斯 渲染后数据的 tensor：包括累积的透射率、最后一个贡献的高斯ID

  // geomFunc、binningFunc、imgFunc 是三个函数指针，其指向的函数可以动态调整内存缓冲区的大小
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0; // 初始化渲染的tile个数为0

  if(P != 0) {
      // 场景中存在高斯，则进行光栅化

      // 如果参数中输入了所有高斯的球谐系数，则 M=每个高斯的球谐系数个数=16；否则 M=0
	  int M = 0;
	  if(sh.size(0) != 0) {
		M = sh.size(1);
      }

      //! 5. 实际的前向传播
      // 返回：所有高斯覆盖的 tile的总个数
      // 更新：用于管理内存缓冲区的三个Tensor：geomBuffer、binningBuffer、imgBuffer
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
        all_map.contiguous().data<float>(),     // 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),   // 输出的 RGB图像，考虑了背景颜色，(3,H,W)
		radii.contiguous().data<int>(),         // 输出的 所有高斯 投影在当前相机图像平面的最大半径 数组，(N,)
        out_observe.contiguous().data<int>(),       // 输出的 所有高斯 渲染时在透射率>0.5之前 对某像素有贡献的 像素个数 数组，(N,)
        out_all_map.contiguous().data<float>(),     // 输出的 5通道tensor，[0-2]：渲染的法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
        out_plane_depth.contiguous().data<float>(), // 输出的 无偏深度图（相机坐标系）
        render_geo,     // 是否要渲染 深度图和法向量图的标志，默认为False
		debug);
  }

  return std::make_tuple(rendered, out_color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer);
}


/**
 * 反向传播，求出
 * @return: 一个元组，包含：  dL_dmeans2D：
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,    // 背景颜色，默认为[0,0,0]，黑色
    const torch::Tensor& all_map_pixels,    // forward输出的 5通道tensor，[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
	const torch::Tensor& means3D,   // 所有高斯 中心的世界坐标
	const torch::Tensor& radii,     // 所有高斯 投影在当前相机图像平面上的最大半径
    const torch::Tensor& colors,    // python代码中 预计算的颜色，默认是空tensor
    const torch::Tensor& all_maps,  // 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
	const torch::Tensor& scales,    // 所有高斯的 缩放因子
	const torch::Tensor& rotations, // 所有高斯的 旋转四元数
	const float scale_modifier, // 缩放因子调节系数
	const torch::Tensor& cov3D_precomp, // python代码中 预计算的3D协方差矩阵，默认为空tensor
	const torch::Tensor& viewmatrix,    // 观测变换矩阵，W2C
    const torch::Tensor& projmatrix,    // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,         // 输入的 loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
    const torch::Tensor& dL_dout_all_map,       // 输入的 loss对forward输出的 5通道tensor（[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离）的梯度
    const torch::Tensor& dL_dout_plane_depth,   // 输入的 loss对渲染的无偏深度图 的梯度
	const torch::Tensor& sh,    // 所有高斯的 球谐系数，(N,16,3)
	const int degree,   // 当前的球谐阶数
	const torch::Tensor& campos,    // 当前相机中心的世界坐标
	const torch::Tensor& geomBuffer,    // 存储所有高斯 几何数据的 tensor：包括2D中心像素坐标、相机坐标系下的深度、3D协方差矩阵等
	const int R,        // 所有高斯覆盖的 tile的总个数
	const torch::Tensor& binningBuffer, // 存储所有高斯 排序数据的 tensor：包括未排序和排序后的 所有高斯覆盖的tile的 keys、values列表
	const torch::Tensor& imageBuffer,   // 存储所有高斯 渲染后数据的 tensor：包括累积的透射率、最后一个贡献的高斯ID
    const bool render_geo,  // 是否要渲染 深度图和法向量图的标志，默认为False
	const bool debug)   // 默认为False
{
  const int P = means3D.size(0);    // 所有高斯的个数
  const int H = dL_dout_color.size(1);  // 图像高
  const int W = dL_dout_color.size(2);  // 图像宽

  // 如果参数中输入了所有高斯的球谐系数，则 M=每个高斯的球谐系数个数=16；否则 M=0
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D_abs = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dall_map = torch::zeros({P, NUM_ALL_MAP}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0) {
      //! 实际反向传播
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
      all_map_pixels.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
      all_maps.contiguous().data<float>(),
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
	  dL_dout_color.contiguous().data<float>(),         // 输入的 loss对渲染的RGB图像中每个像素颜色 的梯度（优化器输出的值，由优化器在训练迭代中自动计算）
      dL_dout_all_map.contiguous().data<float>(),   // 输入的 loss对forward输出的 5通道tensor（[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离）的梯度
      dL_dout_plane_depth.contiguous().data<float>(),   // 输入的 loss对渲染的无偏深度图 的梯度
	  dL_dmeans2D.contiguous().data<float>(),   // 输出的 loss对所有高斯 中心投影到图像平面的像素坐标 的梯度
      dL_dmeans2D_abs.contiguous().data<float>(),   // 输出的 loss对所有高斯 中心2D投影像素坐标 的梯度 的绝对值
	  dL_dconic.contiguous().data<float>(),         // 输出的 loss对所有高斯 2D协方差矩阵 的梯度
	  dL_dopacity.contiguous().data<float>(),   // 输出的 loss对所有高斯 不透明度 的梯度
	  dL_dcolors.contiguous().data<float>(),        // 输出的 loss对所有高斯 在当前相机中心的观测方向下 的RGB颜色值 的梯度
	  dL_dmeans3D.contiguous().data<float>(),   // 输出的 loss对所有高斯 中心世界坐标 的梯度
	  dL_dcov3D.contiguous().data<float>(),     // 输出的 loss对所有高斯 3D协方差矩阵 的梯度
	  dL_dsh.contiguous().data<float>(),        // 输出的 loss对所有高斯 球谐系数 的梯度
	  dL_dscales.contiguous().data<float>(),    // 输出的 loss对所有高斯 缩放因子 的梯度
	  dL_drotations.contiguous().data<float>(),     // 输出的 loss对所有高斯 旋转四元数 的梯度
      dL_dall_map.contiguous().data<float>(),   // 输出的
      render_geo,   // 是否要渲染 深度图和法向量图的标志，默认为False
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}


/**
 * 检查所有高斯是否在当前相机的视锥体内
 * @return: present：所有高斯 是否在当前相机视锥体内的标志 数组
 */
torch::Tensor markVisible(
		torch::Tensor& means3D,     // 所有高斯 中心的世界坐标
		torch::Tensor& viewmatrix,  // 观测变换矩阵，W2C
		torch::Tensor& projmatrix)  // 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
{ 
  const int P = means3D.size(0);    // 所有高斯的个数

  // 输出的 所有高斯 是否在当前相机视锥体内的标志 数组， (N，)
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