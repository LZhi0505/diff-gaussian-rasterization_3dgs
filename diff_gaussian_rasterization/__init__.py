#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


# 调用自定义的_RasterizeGaussians的apply方法，其继承自torch.autograd.Function，需自己写forward和backward，并传递了一系列参数进行高斯光栅化
def rasterize_gaussians(
    means3D,    # 高斯分布的三维坐标
    means2D,    # 高斯分布的二维坐标（屏幕空间坐标）
    sh,         # 球谐系数
    colors_precomp, # 预计算的颜色
    opacities,      # 不透明度
    scales,         # 缩放因子
    rotations,      # 旋转
    cov3Ds_precomp, # 预计算的三维协方差矩阵
    raster_settings,    # 高斯光栅化的设置
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    """
    自定义的、继承自torch.autograd.Function的类，用于高斯光栅化的前向传播和反向传播，需重写forward和backward
    backward函数 和 forward函数输入输出需要相对应：
        forward输出了color, radii，则 backward输入ctx, grad_out_color, 也就是上下文信息和两个forward输出的变量的grad；
        forward输入了除ctx外9个变量，则 backward返回9个梯度，不需要梯度的变量用None占位

    forward 和 backward中调用了_C.rasterize_gaussians 和 _C.rasterize_gaussians_backward，这是C函数，其桥梁在文件./submodules/diff-gaussian-rasterization/ext.cpp中定义；
    即
    """
    @staticmethod
    # 定义前向渲染的规则，调用C++/CUDA实现的 _C.rasterize_gaussians 方法进行高斯光栅化
    # 输入：除ctx外的9个参数
    # 输出：color, radii
    def forward(
        ctx,    # 上下文信息，用于保存计算中间结果以供反向传播使用
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # 按照c++库所期望的形式重构参数
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # 调用C++/CUDA实现的光栅器对3D高斯进行光栅化
        if raster_settings.debug:
            # 若光栅器的设置中开启了调试模式，则在计算前向和反向传播时保存了参数的副本，并在出现异常时将其保存到文件中，以供调试
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 默认，调用C++/CUDA实现的 _C.rasterize_gaussians 方法进行高斯光栅化
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    # 定义反向传播梯度下降的规则。根据前向传播的结果和外部梯度计算输入张量的梯度。
    # 输入：上下文信息、两个forward输出的变量的grad
    # 输出：forward输入的9个变量的梯度，不需要返回梯度的变量用None占位
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 将梯度和其他输入参数重构为 C++ 方法所期望的形式
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # 调用反向传播方法 计算相关tensor的梯度
        if raster_settings.debug:
            # 若光栅器的设置中开启了调试模式，则在计算前向和反向传播时保存了参数的副本，并在出现异常时将其保存到文件中，以供调试
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            # 默认，调用反向传播方法 计算相关tensor的梯度
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # 返回9个参数梯度
        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

# 定义初始化类GaussianRasterizer的配置参数
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

# 光栅器类，继承自nn.Module类，用于光栅化3D高斯
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        # 初始化方法，接受一个raster_settings参数，该参数包含了光栅化的设置（例如图像大小、视场、背景颜色等）
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        """
        基于当前相机的视锥体，返回3D高斯是否在当前相机视野中是否可见的mask（使用C++/CUDA代码执行）
        :param positions: 3D高斯的中心位置
        """
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        """
        前向传播，进行3D高斯光栅化
            means3D: 3D坐标
            means2D: 2D坐标
            opacities:   透明度
            shs:         SH特征
            colors_precomp:  预计算的颜色
            scales:      缩放因子
            rotations:   旋转
            cov3D_precomp:   预计算的3D协方差矩阵
        """
        raster_settings = self.raster_settings

        # 检查SH特征和预计算的颜色是否同时提供，要求只提供其中一种
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        # 检查缩放/旋转对或预计算的3D协方差是否同时提供，要求只提供其中一种
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 如果某个输入参数为None，则将其初始化为空张量
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # 调用C++/CUDA光栅化例程rasterize_gaussians，传递相应的输入参数和光栅化设置。rasterize_gaussians又调用_RasterizeGaussians
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

