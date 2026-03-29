// Copyright (c) 2025 Dmitry Rogozhkin.

#include "ColorConversionKernel.h"
#include <algorithm> // For std::clamp

namespace facebook::torchcodec {

// NV12 to RGB conversion kernel with Bicubic Interpolation
// Output: uint8_t for TorchCodec (0-255)
struct NV12toRGBKernel {
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> y_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> uv_acc;
  sycl::accessor<uint8_t, 1, sycl::access::mode::write> rgb_acc;
  int width;
  int height;
  int stride;
  int color_std;

  NV12toRGBKernel(
      sycl::accessor<uint8_t, 1, sycl::access::mode::read> y_acc,
      sycl::accessor<uint8_t, 1, sycl::access::mode::read> uv_acc,
      sycl::accessor<uint8_t, 1, sycl::access::mode::write> rgb_acc,
      int width,
      int height,
      int stride,
      int color_std)
      : y_acc(y_acc), uv_acc(uv_acc), rgb_acc(rgb_acc), width(width), height(height), stride(stride), color_std(color_std) {}

  void operator()(sycl::id<2> idx) const {
    int x = idx[1];
    int y = idx[0];

    if (x >= width || y >= height) {
      return;
    }

    // --- INTEGER ARITHMETIC IMPLEMENTATION ---
    // Scaled by 16 bits (65536) for precision matching libswscale
    
    // RGB = Y_term + UV_term
    // Y_term = (Y - 16) * CY
    // UV_terms differ by channel
    
    // Constants (Scale = 16 bits)
    // CY = 255/219 * 65536 = 76309
    const int CY = 76309;
    
    int CV_R, CU_G, CV_G, CU_B;
    
    if (color_std == 0) {
        // BT.601 (SD)
        // RGB = Y + Coeff*(Val-128)
        // CV_R = 1.402 * 255/224 * 65536 = 104597
        // CU_G = 0.344136 * 255/224 * 65536 = 25675
        // CV_G = 0.714136 * 255/224 * 65536 = 53281
        // CU_B = 1.772 * 255/224 * 65536 = 132201
        CV_R = 104597;
        CU_G = 25675;
        CV_G = 53281;
        CU_B = 132201;
    } else {
        // BT.709 (HD) - Default
        // CV_R = 1.5748 * 255/224 * 65536 = 117489
        // CU_G = 0.187324 * 255/224 * 65536 = 13975
        // CV_G = 0.468124 * 255/224 * 65536 = 34925
        // CU_B = 1.8556 * 255/224 * 65536 = 138443
        CV_R = 117489;
        CU_G = 13975;
        CV_G = 34925;
        CU_B = 138443;
    }

    // --- Luma (Y) ---
    uint8_t Y_raw = y_acc[y * stride + x];
    int y_term = (Y_raw - 16) * CY; // result is roughly 24 bits

    // --- Chroma (UV) with Bilinear Interpolation (Integer) ---
    // Siting: JPEG/Center (x/2.0 - 0.25)
    // Scale weights by 2048 (11 bits, common in sws)
    
    float u_pos = (x / 2.0f) - 0.25f;
    float v_pos = (y / 2.0f) - 0.25f;
    
    int ux = (int)sycl::floor(u_pos);
    int uy = (int)sycl::floor(v_pos);
    
    // Calculate weights (scale 2048)
    // dx_f = u_pos - ux. 
    // prev code: float dx = u_pos - ux;
    // int w_x = dx * 2048.
    
    int w_x = (int)((u_pos - ux) * 2048.0f);
    int w_y = (int)((v_pos - uy) * 2048.0f);
    
    // Clamp coordinates
    int uv_width = width / 2;
    int uv_height = height / 2;
    
    int ux0 = sycl::clamp(ux, 0, uv_width - 1);
    int uy0 = sycl::clamp(uy, 0, uv_height - 1);
    int ux1 = sycl::clamp(ux + 1, 0, uv_width - 1);
    int uy1 = sycl::clamp(uy + 1, 0, uv_height - 1);
    
    // Fetch UVs
    int off00 = uy0 * stride + ux0 * 2;
    int off10 = uy0 * stride + ux1 * 2;
    int off01 = uy1 * stride + ux0 * 2;
    int off11 = uy1 * stride + ux1 * 2;
    
    int u00 = uv_acc[off00];
    int v00 = uv_acc[off00 + 1];
    int u10 = uv_acc[off10];
    int v10 = uv_acc[off10 + 1];
    int u01 = uv_acc[off01];
    int v01 = uv_acc[off01 + 1];
    int u11 = uv_acc[off11];
    int v11 = uv_acc[off11 + 1];
    
    // Bilinear Interpolation (Integer)
    // (A*(2048-w) + B*w) >> 11
    
    // Horizontal pass
    int u_row0 = (u00 * (2048 - w_x) + u10 * w_x) >> 11;
    int u_row1 = (u01 * (2048 - w_x) + u11 * w_x) >> 11;
    int v_row0 = (v00 * (2048 - w_x) + v10 * w_x) >> 11;
    int v_row1 = (v01 * (2048 - w_x) + v11 * w_x) >> 11;
    
    // Vertical pass
    int u_val = (u_row0 * (2048 - w_y) + u_row1 * w_y) >> 11;
    int v_val = (v_row0 * (2048 - w_y) + v_row1 * w_y) >> 11;
    
    // Shift/Normalize UV relative to 128
    int u_diff = u_val - 128;
    int v_diff = v_val - 128;

    // --- Color Conversion (Integer) ---
    // R = Y + V*CV_R
    int r_val = y_term + v_diff * CV_R;
    
    // G = Y - U*CU_G - V*CV_G
    int g_val = y_term - u_diff * CU_G - v_diff * CV_G;
    
    // B = Y + U*CU_B
    int b_val = y_term + u_diff * CU_B;
    
    // Scale back (>> 16)
    // Add rounding half-bit (1<<15) before shift? sws_scale usually creates 'out' table.
    // Truncation was requested, so straight shift.
    r_val = r_val >> 16;
    g_val = g_val >> 16;
    b_val = b_val >> 16;

    // Clamp
    uint8_t r = (uint8_t)sycl::clamp(r_val, 0, 255);
    uint8_t g = (uint8_t)sycl::clamp(g_val, 0, 255);
    uint8_t b = (uint8_t)sycl::clamp(b_val, 0, 255);

    // Write output
    int rgb_idx = (y * width + x) * 3;
    rgb_acc[rgb_idx + 0] = r;
    rgb_acc[rgb_idx + 1] = g;
    rgb_acc[rgb_idx + 2] = b;
  }
};

void convertNV12ToRGB(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue,
    int color_std) {

  size_t y_size = stride * height;
  size_t uv_size = stride * height / 2;  // NV12: UV is half resolution, interleaved
  size_t rgb_size = width * height * 3;

  // Create SYCL buffers from raw pointers
  sycl::buffer<uint8_t, 1> y_buf(y_plane, sycl::range<1>(y_size));
  sycl::buffer<uint8_t, 1> uv_buf(uv_plane, sycl::range<1>(uv_size));
  sycl::buffer<uint8_t, 1> rgb_buf(rgb_output, sycl::range<1>(rgb_size));

  // Submit kernel to SYCL queue
  queue.submit([&](sycl::handler& cgh) {
    // Get accessors for buffers
    auto y_acc = y_buf.get_access<sycl::access::mode::read>(cgh);
    auto uv_acc = uv_buf.get_access<sycl::access::mode::read>(cgh);
    auto rgb_acc = rgb_buf.get_access<sycl::access::mode::write>(cgh);

    // Create kernel instance
    NV12toRGBKernel kernel(y_acc, uv_acc, rgb_acc, width, height, stride, color_std);

    // Launch 2D kernel (one work-item per pixel)
    cgh.parallel_for(
        sycl::range<2>(height, width),
        kernel);
  });

  // Wait for kernel completion
  queue.wait();
}

// This function is called during library initialization to ensure
// the SYCL runtime registers the kernel associated with this type.
void registerColorConversionKernel() {
  // Creating a dummy pointer to the kernel type is often enough
  // to force the compiler to emit the necessary RTTI/integration info.
  // We use volatile to prevent optimization.
  volatile size_t s = sizeof(NV12toRGBKernel);
  (void)s;
}

} // namespace facebook::torchcodec