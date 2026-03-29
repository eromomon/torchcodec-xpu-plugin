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

// Anchor function to force kernel registration
void registerColorConversionKernel();

} // namespace facebook::torchcodec

