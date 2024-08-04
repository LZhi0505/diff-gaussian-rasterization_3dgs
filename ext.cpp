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

#include <torch/extension.h>
#include "rasterize_points.h"

/**
 * _C.rasterize_gaussians 和 rasterize_points.h中的C函数 RasterizeGaussiansCUDA 绑定；
 * _C.rasterize_gaussians_backward 和 rasterize_points.h中的C函数 RasterizeGaussiansBackwardCUDA 绑定；
 * mark_visible 和 rasterize_points.h中的C函数 markVisible 绑定
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}