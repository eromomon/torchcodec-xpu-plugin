// Copyright (c) 2025 Dmitry Rogozhkin.

#include "ColorConversionKernel.h"
#include <algorithm> // For std::clamp

namespace facebook::torchcodec {

void convertNV12ToRGB(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue,
    int color_std) {

  // USM-based kernel: work directly with device pointers
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
        int x = idx[1];
        int y = idx[0];

        if (x >= width || y >= height) return;

        // --- INTEGER ARITHMETIC IMPLEMENTATION ---
        const int CY = 76309;
        
        int CV_R, CU_G, CV_G, CU_B;
        
        if (color_std == 0) {
            // BT.601 (SD)
            CV_R = 104597;
            CU_G = 25675;
            CV_G = 53281;
            CU_B = 132201;
        } else {
            // BT.709 (HD) - Default
            CV_R = 117489;
            CU_G = 13975;
            CV_G = 34925;
            CU_B = 132201;
        }

        // --- Luma (Y) ---
        uint8_t Y_raw = y_plane[y * stride + x];
        int y_term = (Y_raw - 16) * CY;

        // --- Chroma (UV) with Bilinear Interpolation ---
        float u_pos = (x / 2.0f) - 0.25f;
        float v_pos = (y / 2.0f) - 0.25f;
        
        int ux = (int)sycl::floor(u_pos);
        int uy = (int)sycl::floor(v_pos);
        
        int w_x = (int)((u_pos - ux) * 2048.0f);
        int w_y = (int)((v_pos - uy) * 2048.0f);
        
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
        
        int u00 = uv_plane[off00];
        int v00 = uv_plane[off00 + 1];
        int u10 = uv_plane[off10];
        int v10 = uv_plane[off10 + 1];
        int u01 = uv_plane[off01];
        int v01 = uv_plane[off01 + 1];
        int u11 = uv_plane[off11];
        int v11 = uv_plane[off11 + 1];
        
        // Bilinear Interpolation
        int u_row0 = (u00 * (2048 - w_x) + u10 * w_x) >> 11;
        int u_row1 = (u01 * (2048 - w_x) + u11 * w_x) >> 11;
        int v_row0 = (v00 * (2048 - w_x) + v10 * w_x) >> 11;
        int v_row1 = (v01 * (2048 - w_x) + v11 * w_x) >> 11;
        
        int u_val = (u_row0 * (2048 - w_y) + u_row1 * w_y) >> 11;
        int v_val = (v_row0 * (2048 - w_y) + v_row1 * w_y) >> 11;
        
        int u_diff = u_val - 128;
        int v_diff = v_val - 128;

        // --- Color Conversion ---
        int r_val = (y_term + v_diff * CV_R) >> 16;
        int g_val = (y_term - u_diff * CU_G - v_diff * CV_G) >> 16;
        int b_val = (y_term + u_diff * CU_B) >> 16;

        // Clamp
        uint8_t r = (uint8_t)sycl::clamp(r_val, 0, 255);
        uint8_t g = (uint8_t)sycl::clamp(g_val, 0, 255);
        uint8_t b = (uint8_t)sycl::clamp(b_val, 0, 255);

        // Write output
        int rgb_idx = (y * width + x) * 3;
        rgb_output[rgb_idx + 0] = r;
        rgb_output[rgb_idx + 1] = g;
        rgb_output[rgb_idx + 2] = b;
    });
  });

  queue.wait();
}

void detachTiledNV12(
    const uint8_t* tiled_y_plane,
    const uint8_t* tiled_uv_plane,
    uint8_t* linear_y_output,
    uint8_t* linear_uv_output,
    int width,
    int height,
    int stride,
    sycl::queue& queue) {

    // USM-based kernel: work directly with device pointers
    // Y plane dimensions: width × height
    // UV plane dimensions: width × (height/2) with interleaved U,V samples
        
    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            int x = idx[1];
            int y = idx[0];

            if (x >= width || y >= height) return;

            // Helper lambda for Intel Tile-Y offset calculation
            // Intel Y-Tiling uses COLUMN-MAJOR OWord organization
            // Tile: 128 bytes wide × 32 rows = 4KB
            // Within tile: 8 OWords (16-byte columns) arranged column-by-column
            // Each OWord covers all 32 rows before moving to next OWord
            auto get_offset_tile_y = [](int x, int y, int stride) -> size_t {
                const int TileW = 128;  // Tile width in bytes
                const int TileH = 32;   // Tile height in rows
                const int OWordSize = 16; // OWord = 16 bytes
                const int TileSize = TileW * TileH;  // 4096 bytes per tile
                
                // Which tile does this pixel belong to?
                int tile_x = x / TileW;
                int tile_y = y / TileH;
                
                // Position within the tile
                int x_in_tile = x % TileW;
                int y_in_tile = y % TileH;
                

                // Block position added to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
                int block_x = x_in_tile / 64;  // width of pixel blocks
                int block_y = y_in_tile / 4;   // heigh of pixel blocks



                // Y-Tiling: Column-major OWord layout
                // OWord index (0-7): which 16-byte column within the tile
                int oword_idx = x_in_tile / OWordSize;
                // Offset within OWord (0-15)
                int offset_in_oword = x_in_tile % OWordSize;

                int sub_tile_size = OWordSize * 4;
                int sub_tile_y = y_in_tile / 4;
                int y_in_sub_tile = y_in_tile % 4;

                // conditional to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
                if ((block_x ^ block_y ) & 0x1){
                    block_x ^= 1;
                    block_y ^= 1;

                    x_in_tile = block_x * 64 + (x_in_tile % 64);
                    y_in_tile = block_y * 4 + (y_in_tile % 4);

                    sub_tile_y = block_y;
                    y_in_sub_tile = y_in_tile % 4;

                    oword_idx = x_in_tile / OWordSize;
                    offset_in_oword = x_in_tile % 16;

                }
                
                int offset_in_tile = (sub_tile_y * TileW/OWordSize + oword_idx) * sub_tile_size + y_in_sub_tile * OWordSize + offset_in_oword;

                // Number of tiles per row
                int stride_in_tiles = stride / TileW;
                
                // Final tiled offset
                size_t tile_offset = (size_t)(tile_y * stride_in_tiles + tile_x) * TileSize;
                return tile_offset + offset_in_tile;
            };

            // Detile Y Plane
            size_t linear_idx_y = (size_t)y * stride + x;
            size_t tiled_idx_y = get_offset_tile_y(x, y, stride);
            linear_y_output[linear_idx_y] = tiled_y_plane[tiled_idx_y];

            // Detile UV Plane (half height for NV12)
            // UV samples are interleaved: U0,V0,U1,V1,... in a row
            if (y < height / 2) {
                size_t linear_idx_uv = (size_t)y * stride + x;
                size_t tiled_idx_uv = get_offset_tile_y(x, y, stride);
                linear_uv_output[linear_idx_uv] = tiled_uv_plane[tiled_idx_uv];
            }
        });
    });

    queue.wait();
    
}

} // namespace facebook::torchcodec