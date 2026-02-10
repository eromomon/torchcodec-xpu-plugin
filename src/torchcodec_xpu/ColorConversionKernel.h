// Copyright (c) 2025 Dmitry Rogozhkin.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace facebook::torchcodec {

// High-level conversion function for NV12 to RGB
// Works with raw pointers for maximum flexibility
void convertNV12ToRGB(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue,
    int color_std = 1); // 0 = BT.601, 1 = BT.709

// Detiling function: Tiled NV12 -> Linear NV12
// This runs before color conversion if the input is tiled.
void detachTiledNV12(
    const uint8_t* tiled_y_plane,
    const uint8_t* tiled_uv_plane,
    uint8_t* linear_y_output,
    uint8_t* linear_uv_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue);

// Anchor function to force kernel registration
void registerColorConversionKernel();

} // namespace facebook::torchcodec

