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


# 调用自定义的_RasterizeGaussians的apply方法
def rasterize_gaussians(
    means3D,
    means2D,    # 输出的 所有高斯中心投影在图像平面的坐标
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,    # 光栅化的相关设置（图像尺寸、视场角、背景颜色、变换矩阵、秋谐阶数、相机中心世界坐标等）
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
    该类继承自torch.autograd.Function自动微分类，所以需重写 forward和 backward，用于高斯光栅化的前向传播和反向传播
    forward：自定义操作执行计算，输入参数，返回输出
    backward：如何计算自定义操作相对于其输入的梯度，根据前向传播的结果和外部梯度计算输入张量的梯度

    forward和 backward函数的输入输出需要相对应：
        forward输出了color, radii，则 backward输入ctx, grad_out_color, 也就是上下文信息和两个forward输出的变量的grad；
        forward输入了除ctx外9个变量，则 backward返回9个梯度，不需要梯度的变量用None占位

    forward和 backward中分别调用了 _C.rasterize_gaussians和 _C.rasterize_gaussians_backward，这是C函数，其桥梁在文件./submodules/diff-gaussian-rasterization/ext.cpp中定义；
    即
    """
    @staticmethod
    # 定义前向渲染的规则，调用C++/CUDA实现的 _C.rasterize_gaussians方法进行高斯光栅化
    # 输入：除ctx外的9个参数
    # 输出：color, radii
    def forward(
        ctx,    # 上下文信息，用于保存计算中间结果以供反向传播使用
        means3D,
        means2D,    # 输出的 所有高斯中心投影在图像平面的坐标
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # 1. 按照c++库所期望的形式重构参数，存到一个元组中
        args = (
            raster_settings.bg, # 背景颜色，默认为[1,1,1]，黑色
            means3D,            # 所有高斯 中心的世界坐标
            colors_precomp,     # 预计算的颜色，默认是空tensor，后续在光栅化预处理阶段计算
            opacities,      # 所有高斯的 不透明度
            scales,         # 所有高斯的 缩放因子
            rotations,      # 所有高斯的 旋转四元数
            raster_settings.scale_modifier, # 缩放因子调节系数
            cov3Ds_precomp,                 # 预计算的3D协方差矩阵，默认为空tensor，后续在光栅化预处理阶段计算
            raster_settings.viewmatrix,     # 观测变换矩阵，W2C
            raster_settings.projmatrix,     # 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,                 # 所有高斯的 球谐系数，(N,16,3)
            raster_settings.sh_degree,      # 当前的球谐阶数
            raster_settings.campos,         # 当前相机中心的世界坐标
            raster_settings.prefiltered,    # 预滤除的标志，默认为False
            raster_settings.debug           # 默认为False
        )

        # 2. 光栅化
        if raster_settings.debug:
            # 2.1 若光栅器的设置中开启了调试模式，则在计算前向和反向传播时保存了参数的副本，并在出现异常时将其保存到文件中，以供调试
            cpu_args = cpu_deep_copy_tuple(args) # 先深拷贝一份输入参数
            try:
                # 调用C++/CUDA实现的 _C.rasterize_gaussians()进行光栅化
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 2.2 默认，不是debug模式，则直接调用 _C.rasterize_gaussians()进行光栅化，并传入重构到一个元组中的参数。这个函数与 rasterize_points.cu/ RasterizeGaussiansCUDA绑定
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # 记录前向传播输出的tensor到ctx中，用于反向传播
        ctx.raster_settings = raster_settings   # 光栅化输入参数
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
    viewmatrix : torch.Tensor   # 观测变换矩阵，W2C
    projmatrix : torch.Tensor   # 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
    sh_degree : int         # 当前的球谐阶数
    campos : torch.Tensor   # 当前相机中心的世界坐标
    prefiltered : bool      # 预滤除的标志，默认为False
    debug : bool

# 光栅器类，继承自nn.Module类，用于光栅化3D高斯
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        # 初始化方法，接受一个raster_settings参数，该参数包含了光栅化的设置（如上）
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        """
        基于当前相机的视锥体，返回3D高斯是否在当前相机视野中是否可见的mask（使用C++/CUDA代码执行）
            positions: 高斯的中心位置
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
            means3D: 所有高斯中心的 世界坐标
            means2D: 输出的 所有高斯中心投影在图像平面的坐标
            opacities:   所有高斯的 不透明度
            shs:         所有高斯的 球谐系数，(N,16,3)
            colors_precomp:  预计算的颜色，默认为None，表明在光栅化预处理阶段计算
            scales:      所有高斯的 缩放因子
            rotations:   所有高斯的 旋转四元数
            cov3D_precomp:   预计算的3D协方差矩阵，默认为None，表明在光栅化预处理阶段计算
        """
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            #  检查 球谐系数 和 预计算的颜色值 只能同时提供一种
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            # 检查 缩放/旋转 和 预计算的3D协方差 只能同时提供一种
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 如果某个输入参数为None，则将其初始化为空tensor
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

        # 调用C++/CUDA光栅化例程rasterize_gaussians，传递相应的输入参数和光栅化设置
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

