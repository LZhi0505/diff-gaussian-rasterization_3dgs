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
    extra_feats,    # 默认为空tensor
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
        extra_feats,    # 默认为空tensor
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
        forward输出了color, radii，则 backward输入 上下文信息 ctx, forward输出的 渲染的RGB图像 的梯度 grad_out_color
        forward输入了除ctx外9个变量，则 backward返回9个梯度，不需要梯度的变量用None占位

    在 backward 返回梯度时，需要确保返回的梯度数量与输入变量数量一致

    forward和 backward中分别调用了 _C.rasterize_gaussians和 _C.rasterize_gaussians_backward，这是C函数，其桥梁在文件./submodules/diff-gaussian-rasterization/ext.cpp中定义；
    即
    """
    @staticmethod
    # 定义前向渲染的规则，调用C++/CUDA实现的 _C.rasterize_gaussians方法进行高斯光栅化
    # 输入：除ctx外的9个参数
    # 输出：渲染的RGB图像 color、每个高斯投影在当前相机图像平面上的最大半径 数组 radii
    def forward(
        ctx,    # 上下文信息，用于保存计算中间结果以供反向传播使用
        means3D,
        means2D,    # 输出的 所有高斯中心投影在图像平面的坐标
        sh,
        colors_precomp,
        extra_feats,    # 默认为空tensor
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # 1. 按照c++库所期望的形式重构参数，存到一个元组中
        args = (
            raster_settings.bg, # 背景颜色，默认为[0,0,0]，黑色
            means3D,            # 所有高斯 中心的世界坐标
            colors_precomp,     # 预计算的颜色，默认是空tensor，后续在光栅化预处理阶段计算
            extra_feats,    # 默认为空tensor
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
            raster_settings.record_transmittance,   # 是否返回 所有高斯贡献度 的标志。（只在要使用 贡献度剪枝 时才为True）
            raster_settings.debug           # 默认为False
        )

        # 2. 光栅化（前向传播）
        # 调用 _C.rasterize_gaussians()  --  rasterize_points.cu/ RasterizeGaussiansCUDA 进行光栅化。并传入重构到一个元组中的参数
        # 返回：
        #   所有高斯覆盖的 tile的总个数
        #   渲染的RGB图像
        #   根据额外信息进行a-blending计算的输出，默认为空
        #   透射率接近0.5时的 深度图
        #   每个高斯投影在当前相机图像平面上的最大半径 数组
        #   存储所有高斯的 几何、排序、渲染后数据的tensor
        #   所有高斯 对当前图像有贡献的像素 的贡献度之和
        #   所有高斯 对当前图像有贡献的像素 的个数
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # 先深拷贝一份输入参数，在出现异常时将其保存到文件中，以供调试
            try:
                # 调用C++/CUDA实现的 _C.rasterize_gaussians()进行光栅化
                num_rendered, color, out_extra_feats, median_depth, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_covered_pixels = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 不是debug模式（默认）
            num_rendered, color, out_extra_feats, median_depth, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_covered_pixels = _C.rasterize_gaussians(*args)

        # 如果设置了 返回累计透射率，则只返回：
        #   所有高斯 对当前图像有贡献的像素 的贡献度之和
        #   所有高斯 对当前图像有贡献的像素 的个数
        #   所有高斯 投影在当前相机图像平面上的最大半径 数组
        if raster_settings.record_transmittance:
            return transmittance, num_covered_pixels, radii

        # 3. 记录前向传播输出的tensor到ctx中，用于反向传播
        ctx.raster_settings = raster_settings   # 光栅化输入参数
        ctx.num_rendered = num_rendered     # 渲染的tile的个数
        ctx.save_for_backward(colors_precomp, extra_feats, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)

        # 返回：渲染的RGB图、根据额外信息进行a-blending计算的输出（默认为空）、透射率接近0.5时的 深度图、每个高斯投影在当前相机图像平面上的最大半径 数组
        return color, out_extra_feats, median_depth, radii

    @staticmethod
    # 定义反向传播梯度下降的规则。调用C++/CUDA实现的 _C.rasterize_gaussians_backward方法
    # 输入：上下文信息ctx、loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
    # 输出：loss对forward输入的9个变量（所有高斯的 中心投影到图像平面的像素坐标、中心世界坐标、椭圆二次型矩阵、不透明度、当前相机中心观测下高斯的RGB颜色、球谐系数、3D协方差矩阵、缩放因子、旋转四元数）的梯度，不需要返回梯度的变量用None占位
    def backward(
            ctx,    # 上下文信息
            grad_out_color, # loss对渲染的RGB图像中每个像素颜色的 梯度
            grad_out_extra_feats,
            grad_out_median_depth,
            grad_radii):

        # 从ctx中恢复前向传播输出的tensor
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        assert not raster_settings.record_transmittance, 'should not execute backward for calculate transmittance'
        colors_precomp, extra_feats, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 1. 将梯度和其他输入参数重构为 C++ 方法所期望的形式
        args = (raster_settings.bg, # 背景颜色，默认为[0,0,0]，黑色
                means3D,    # 所有高斯 中心的世界坐标
                radii,      # 所有高斯 投影在当前相机图像平面上的最大半径
                colors_precomp, # python代码中 预计算的颜色，默认是空tensor
                extra_feats,
                scales,     # 所有高斯的 缩放因子
                rotations,  # 所有高斯的 旋转四元数
                raster_settings.scale_modifier,     # 缩放因子调节系数
                cov3Ds_precomp,         # python代码中 预计算的3D协方差矩阵，默认为空tensor
                raster_settings.viewmatrix,     # 观测变换矩阵，W2C
                raster_settings.projmatrix,     # 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,     # 输入的 loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
                grad_out_extra_feats,
                sh,         # 所有高斯的 球谐系数，(N,16,3)
                raster_settings.sh_degree,  # 当前的球谐阶数
                raster_settings.campos,     # 当前相机中心的世界坐标
                geomBuffer,     # 存储所有高斯 几何数据的 tensor：包括2D中心像素坐标、相机坐标系下的深度、3D协方差矩阵等
                num_rendered,   # 所有高斯覆盖的 tile的总个数
                binningBuffer,  # 存储所有高斯 排序数据的 tensor：包括未排序和排序后的 所有高斯覆盖的tile的 keys、values列表
                imgBuffer,      # 存储所有高斯 渲染后数据的 tensor：包括累积的透射率、最后一个贡献的高斯ID
                raster_settings.debug)  # 默认为False

        # 2. 反向传播：计算相关tensor的梯度
        # 调用 _C.rasterize_gaussians_backward()  --  rasterize_points.cu/ RasterizeGaussiansBackwardCUDA 进行反向传播。并传入重构到一个元组中的参数
        # 返回：loss对所有高斯的（中心2D投影像素坐标、观测的RGB颜色、不透明度、中心3D世界坐标、3D协方差矩阵、缩放因子、球谐系数、旋转四元数）的梯度
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # 先深拷贝一份输入参数，在出现异常时将其保存到文件中，以供调试
            try:
                grad_means2D, grad_colors_precomp, grad_pc_extra_feats, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            # 不是debug模式（默认）
            grad_means2D, grad_colors_precomp, grad_pc_extra_feats, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # 返回loss对各参数的梯度
        grads = (
            grad_means3D,   # loss对所有高斯 中心3D世界坐标 的梯度
            grad_means2D,   # loss对所有高斯 中心2D投影像素坐标 的梯度
            grad_sh,        # loss对所有高斯 球谐系数 的梯度
            grad_colors_precomp,    # loss对所有高斯 在当前相机观测下的 RGB颜色值 的梯度
            grad_pc_extra_feats,
            grad_opacities,         # loss对所有高斯 不透明度 的梯度
            grad_scales,            # loss对所有高斯 缩放因子 的梯度
            grad_rotations,         # loss对所有高斯 旋转四元数 的梯度
            grad_cov3Ds_precomp,    # loss对所有高斯 3D协方差矩阵 的梯度
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
    record_transmittance: bool  # 是否返回 所有高斯贡献度 的标志。（只在要使用 贡献度剪枝 时才为True）
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

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, extra_feats = None, scales = None, rotations = None, cov3D_precomp = None):
        """
        前向传播，进行3D高斯光栅化
            means3D: 所有高斯中心的 世界坐标
            means2D: 输出的 所有高斯中心投影在图像平面的坐标
            opacities:   所有高斯的 不透明度
            shs:         所有高斯的 球谐系数，(N,16,3)
            colors_precomp:  预计算的颜色，默认为None，表明在光栅化预处理阶段计算
            extra_feats: 默认为 None
            scales:      所有高斯的 缩放因子
            rotations:   所有高斯的 旋转四元数
            cov3D_precomp:   预计算的3D协方差矩阵，默认为None，表明在光栅化预处理阶段计算
        """
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            # 检查 球谐系数 和 预计算的颜色值 只能同时提供一种
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            # 检查 缩放/旋转 和 预计算的3D协方差 只能同时提供一种
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # 如果某个输入参数为None，则将其初始化为空tensor
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if extra_feats is None:
            extra_feats = torch.Tensor([])

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
            extra_feats,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

