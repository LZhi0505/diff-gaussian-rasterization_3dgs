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
    means2D,    # 为计算 所有高斯中心投影在图像平面的坐标梯度 的占位参数
    means2D_abs,    # 为计算 所有高斯中心投影在图像平面的坐标梯度 绝对值 的占位参数
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    all_map,        # 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
    raster_settings,    # 光栅化的相关设置（图像尺寸、视场角、背景颜色、变换矩阵、秋谐阶数、相机中心世界坐标等）
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_map,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    """
    该类继承自torch.autograd.Function自动微分类，所以需重写 forward和 backward，用于高斯光栅化的前向传播和反向传播
    forward：自定义操作执行计算，输入参数，返回输出
    backward：如何计算自定义操作相对于其输入的梯度，根据前向传播的结果和外部梯度计算输入张量的梯度

    forward的输出 与 backward的输入 要对应，backward的输出 与 forward的输入 要对应：
        forward输出了color, radii，则 backward输入 上下文信息 ctx, forward输出的 渲染的RGB图像 的梯度 grad_out_color
        forward输入了除ctx外9个变量，则 backward返回9个梯度，不需要梯度的变量用None占位

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
        means2D,    # 为计算 所有高斯中心投影在图像平面的坐标梯度 的占位参数
        means2D_abs,    # 为计算 所有高斯中心投影在图像平面的坐标梯度 绝对值 的占位参数
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_maps,   # 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
        raster_settings,
    ):

        # 1. 按照c++库所期望的形式重构参数，存到一个元组中
        args = (
            raster_settings.bg, # 背景颜色，默认为[0,0,0]，黑色
            means3D,            # 所有高斯 中心的世界坐标
            colors_precomp,     # 预计算的颜色，默认是空tensor，后续在光栅化预处理阶段计算
            opacities,      # 所有高斯的 不透明度
            scales,         # 所有高斯的 缩放因子
            rotations,      # 所有高斯的 旋转四元数
            raster_settings.scale_modifier, # 缩放因子调节系数
            cov3Ds_precomp,                 # 预计算的3D协方差矩阵，默认为空tensor，后续在光栅化预处理阶段计算
            all_maps,       # 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
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
            raster_settings.render_geo,     # 是否要渲染 深度图和法向量图的标志，默认为False
            raster_settings.debug           # 默认为False
        )

        # 2. 光栅化（前向传播）
        # 调用 _C.rasterize_gaussians()  --  rasterize_points.cu/ RasterizeGaussiansCUDA 进行光栅化。并传入重构到一个元组中的参数
        # 返回：
        #   所有高斯覆盖的 tile的总个数
        #   渲染的 RGB图像
        #   所有高斯投影在当前相机图像平面上的最大半径 数组
        #   所有高斯 渲染时在透射率>0.5之前 对某像素有贡献的 像素个数 数组
        #   输出的 5通道tensor，[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
        #   渲染的 无偏深度图（相机坐标系）
        #   存储所有高斯的 几何、排序、渲染后数据的tensor
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # 先深拷贝一份输入参数，在出现异常时将其保存到文件中，以供调试
            try:
                num_rendered, color, radii, out_observe, out_all_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 不是debug模式（默认）
            num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # 3. 记录前向传播输出的tensor到ctx中，用于反向传播
        ctx.raster_settings = raster_settings   # 光栅化输入参数
        ctx.num_rendered = num_rendered     # 渲染的tile的个数
        ctx.save_for_backward(out_all_map, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)

        return color, radii, out_observe, out_all_map, out_plane_depth

    @staticmethod
    # 定义反向传播梯度下降的规则。调用C++/CUDA实现的 _C.rasterize_gaussians_backward方法
    # 输入：上下文信息ctx、loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
    # 输出：loss对forward输入的9个变量（所有高斯的 中心3D世界坐标、中心2D投影像素坐标、球谐系数、在当前相机观测下的 RGB颜色值、不透明度、缩放因子、旋转四元数、3D协方差矩阵）的梯度，不需要返回梯度的变量用None占位
    def backward(
            ctx,    # 上下文信息
            grad_out_color, # loss对渲染的RGB图像中每个像素颜色 的梯度
            grad_radii,     # loss对所有高斯投影在当前相机图像平面上的最大半径 的梯度
            grad_out_observe,   # loss对所有高斯 渲染时在透射率>0.5之前 对某像素有贡献的 像素个数 的梯度
            grad_out_all_map,   # loss对forward输出的 5通道tensor 的梯度
            grad_out_plane_depth,   # loss对渲染的无偏深度图 的梯度
            ):

        # 从ctx中恢复前向传播输出的tensor
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        all_map_pixels, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # 1. 将梯度和其他输入参数重构为 C++ 方法所期望的形式
        args = (raster_settings.bg, # 背景颜色，默认为[0,0,0]，黑色
                all_map_pixels,     # forward输出的 5通道tensor，[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离
                means3D,    # 所有高斯 中心的世界坐标
                radii,      # 所有高斯 投影在当前相机图像平面上的最大半径
                colors_precomp, # python代码中 预计算的颜色，默认是空tensor
                all_maps,   # 输入的 5通道tensor，[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 相机光心 到 所有高斯法向量垂直平面的 距离
                scales,     # 所有高斯的 缩放因子
                rotations,  # 所有高斯的 旋转四元数
                raster_settings.scale_modifier,     # 缩放因子调节系数
                cov3Ds_precomp,         # python代码中 预计算的3D协方差矩阵，默认为空tensor
                raster_settings.viewmatrix,     # 观测变换矩阵，W2C
                raster_settings.projmatrix,     # 观测变换*投影变换矩阵，W2NDC = W2C * C2NDC
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,         # 输入的 loss对渲染的RGB图像中每个像素颜色的 梯度（优化器输出的值，由优化器在训练迭代中自动计算）
                grad_out_all_map,       # 输入的 loss对forward输出的 5通道tensor（[0-2]：渲染的 法向量（相机坐标系）；[3]：每个像素对应的 对其渲染有贡献的 所有高斯累加的贡献度；[4]：渲染的 相机光心 到 每个像素穿过的所有高斯法向量垂直平面的 距离）的梯度
                grad_out_plane_depth,   # 输入的 loss对 渲染的无偏深度图 的梯度
                sh,         # 所有高斯的 球谐系数，(N,16,3)
                raster_settings.sh_degree,  # 当前的球谐阶数
                raster_settings.campos,     # 当前相机中心的世界坐标
                geomBuffer,     # 存储所有高斯 几何数据的 tensor：包括2D中心像素坐标、相机坐标系下的深度、3D协方差矩阵等
                num_rendered,   # 所有高斯覆盖的 tile的总个数
                binningBuffer,  # 存储所有高斯 排序数据的 tensor：包括未排序和排序后的 所有高斯覆盖的tile的 keys、values列表
                imgBuffer,      # 存储所有高斯 渲染后数据的 tensor：包括累积的透射率、最后一个贡献的高斯ID
                raster_settings.render_geo, # 是否要渲染 深度图和法向量图的标志，默认为False
                raster_settings.debug)  # 默认为False

        # 2. 反向传播：计算相关tensor的梯度
        # 调用 _C.rasterize_gaussians_backward()  --  rasterize_points.cu/ RasterizeGaussiansBackwardCUDA 进行反向传播。并传入重构到一个元组中的参数
        # 返回：
        #   loss对所有高斯 中心2D投影像素坐标 的梯度
        #   loss对所有高斯 中心2D投影像素坐标 的梯度 的绝对值
        #   loss对所有高斯 在当前相机中心的观测方向下 的RGB颜色值 的梯度
        #   loss对所有高斯 不透明度 的梯度
        #   loss对所有高斯 中心3D世界坐标 的梯度
        #   loss对所有高斯 3D协方差矩阵 的梯度
        #   loss对所有高斯 球谐系数 的梯度
        #   loss对所有高斯 缩放因子 的梯度
        #   loss对所有高斯 旋转四元数 的梯度
        #   loss对所有高斯 5通道tensor（法向量、贡献度、光心到高斯法切平面距离）的梯度
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # 先深拷贝一份输入参数，在出现异常时将其保存到文件中，以供调试
            try:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            # 不是debug模式（默认）
            grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)

        # 返回loss对各参数的梯度
        grads = (
            grad_means3D,   # loss对所有高斯 中心3D世界坐标 的梯度
            grad_means2D,   # loss对所有高斯 中心2D投影像素坐标 的梯度
            grad_means2D_abs,   # loss对所有高斯 中心2D投影像素坐标 的梯度 的绝对值
            grad_sh,        # loss对所有高斯 球谐系数 的梯度
            grad_colors_precomp,    # loss对所有高斯 在当前相机观测下的 RGB颜色值 的梯度
            grad_opacities,         # loss对所有高斯 不透明度 的梯度
            grad_scales,            # loss对所有高斯 缩放因子 的梯度
            grad_rotations,         # loss对所有高斯 旋转四元数 的梯度
            grad_cov3Ds_precomp,    # loss对所有高斯 3D协方差矩阵 的梯度
            gard_all_map,           # loss对所有高斯 5通道tensor（法向量、贡献度、光心到高斯法切平面距离）的梯度
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
    render_geo : bool       # 是否要渲染 深度图和法向量图的标志，默认为False
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

    def forward(self, means3D, means2D, means2D_abs, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, all_map=None):
        """
        前向传播，进行3D高斯光栅化
            means3D: 所有高斯中心的 世界坐标
            means2D: 输出的 所有高斯中心投影在图像平面的坐标
            means2D_abs: 输出的 所有高斯中心投影在图像平面的坐标 的绝对值
            opacities:   所有高斯的 不透明度
            shs:         所有高斯的 球谐系数，(N,16,3)
            colors_precomp:  预计算的颜色，默认为None，表明在光栅化预处理阶段计算
            scales:      所有高斯的 缩放因子
            rotations:   所有高斯的 旋转四元数
            cov3D_precomp:   预计算的3D协方差矩阵，默认为None，表明在光栅化预处理阶段计算
            all_map:     输入、输出的tensor。输入时为[0-2]: 当前相机坐标系下所有高斯的法向量，即最短轴向量；[3]: 全1.0；[4]: 所有高斯中心沿其法向量方向 与 相机光心的距离投影
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

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if all_map is None:
            all_map = torch.Tensor([])

        # 调用C++/CUDA光栅化例程rasterize_gaussians，传递相应的输入参数和光栅化设置
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            all_map,
            raster_settings, 
        )

