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
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;        // 存储用于排序操作的缓冲区大小
		uint64_t* point_list_keys_unsorted; // 未排序的 key列表，[tile ID | 3D高斯的深度]
		uint64_t* point_list_keys;          // 排序后的 key列表
		uint32_t* point_list_unsorted;  // 未排序的 value列表，[3D高斯的 ID]
		uint32_t* point_list;           // 排序后的 value列表
		char* list_sorting_space;   // 用于排序操作的缓冲区

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