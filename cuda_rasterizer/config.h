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

/**
 * 记录了render的颜色通道数和BLOCK的长宽。
 * 所以其实要render多种信息（比如深度，normal，透明度etc.）只要修改颜色通道数就可以并行render了，不用花串行那个时间，说的就是GaussianShader，很蠢
 *
 * 只是修改常数值or函数头，也可以还有其他一些，总之如果不修改函数体的话，重新编译可能并不会有效。
 * 可以随便在函数体里面加一句printf("haha")之类的，编译然后再删掉然后再编译，才能起效果。比如修改config.h的颜色通道数就需要这样重新编译。
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_ALL_MAP 5
#define BLOCK_X 16
#define BLOCK_Y 16

#endif