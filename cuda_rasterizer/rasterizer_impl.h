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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
    /**
     * 从一个动态分配的内存块（chunk）中为不同类型的数组（ptr）分配空间
     * @param chunk 输入/输出参数，指向动态分配的内存块的起始地址。函数执行后，chunk会被更新为下一个可用的内存块地址
     * @param ptr   输出参数，一个指向分配数组的指针引用
     * @param count 数组中元素的数量
     * @param alignment 内存对齐的字节数，通常为 128字节
     */
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);  // 计算以 alignment=128字节对齐的 下一个对齐边界的地址
		ptr = reinterpret_cast<T*>(offset);     // 将对齐后的地址强制转换为目标类型 T*,赋给 ptr 指针
		chunk = reinterpret_cast<char*>(ptr + count);   // 将 chunk 指针向前移动,使其指向下一个可用的内存块起始位置
	}

    // 存储所有高斯的各个参数的结构体
	struct GeometryState
	{
		size_t scan_size; // 临时显存空间的大小
		float* depths;      // 所有高斯 中心在当前相机坐标系下的z值 数组
		char* scanning_space; // 额外需要的临时显存空间
		bool* clamped;      // 所有高斯 是否被裁剪的标志 数组，某位置为 True表示：该高斯在当前相机的观测角度下，其RGB值3个的某个值 < 0，在后续渲染中不考虑它
		int* internal_radii;
		float2* means2D;    // 所有高斯 中心投影在当前相机图像平面的二维坐标 数组
		float* cov3D;       // 所有高斯 在世界坐标系下的3D协方差矩阵 数组
		float4* conic_opacity;  // 所有高斯 2D协方差的逆 和 不透明度 数组
		float* rgb;         // 所有高斯 在当前相机中心的观测方向下 的RGB颜色值 数组
		uint32_t* point_offsets;    // 所有高斯 覆盖的 tile个数的 前缀和 数组，每个元素是 从第一个高斯到当前高斯所覆盖的所有tile的数量
		uint32_t* tiles_touched;    // 所有高斯 在当前相机图像平面覆盖的线程块 tile的个数 数组

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;  // 每个tile在 排序后的keys列表中的 起始和终止位置。索引：tile_ID；值[x,y)：该tile在keys列表中起始、终止位置，个数y-x：落在该tile_ID上的高斯的个数
		uint32_t* n_contrib;    // 渲染每个像素 pixel穿过的高斯的个数，也是最后一个对渲染该像素RGB值 有贡献的高斯ID 的数组
		float* accum_alpha;     // 渲染后每个像素 pixel的 累积的透射率 的数组

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size; // 临时显存空间的大小
		uint64_t* point_list_keys_unsorted; // 未排序的 所有高斯覆盖的tile的 keys列表，分布顺序：大顺序：各高斯，小顺序：该高斯覆盖的各tile_ID。每个元素是 (tile_ID | 3D高斯的深度)
		uint64_t* point_list_keys;          // 排序后的 keys列表，分布顺序：大顺序：各tile_ID，小顺序：落在该tile内各高斯的深度

        uint32_t* point_list_unsorted;  // 未排序的 所有高斯覆盖的tile的 values列表，每个元素是 对应高斯的ID
		uint32_t* point_list;           // 排序后的 value列表
		char* list_sorting_space; // 排序时用到的临时显存空间

		static BinningState fromChunk(char*& chunk, size_t P);
	};


    // 计算存储 T 类型数据所需的内存大小的函数
    // 通过调用 T::fromChunk 并传递一个空指针（nullptr）来模拟内存分配过程
    // 通过这个过程，它确定了实际所需的内存大小，加上额外的 128 字节以满足可能的内存对齐要求
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};