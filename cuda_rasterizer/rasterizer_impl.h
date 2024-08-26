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
     * 从一个动态分配的内存块中提取并初始化指定类型的数组指针
     * @tparam T
     * @param chunk 输入/输出参数,指向动态分配的内存块的起始地址。函数执行后，chunk会被更新为下一个可用的内存块地址
     * @param ptr   输出参数,初始化为指向内存块中指定类型 T数组的指针
     * @param count 要初始化的 T 类型元素的个数
     * @param alignment 内存对齐的字节数，通常为 128字节
     */
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);  // 计算以 alignment=128字节对齐的 下一个对齐边界的地址
		ptr = reinterpret_cast<T*>(offset);     // 将对齐后的地址强制转换为目标类型 T*,赋给 ptr 指针
		chunk = reinterpret_cast<char*>(ptr + count);   // 将 chunk 指针向前移动,使其指向下一个可用的内存块起始位置
	}

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
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};