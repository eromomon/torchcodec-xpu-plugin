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

// Detiling Kernel implementation
// Checks addressing for TileY (Gen12/Xe) or Tile4 (Xe2/Arc) ideally,
// but here we provide the structural skeleton.
struct DetileNV12Kernel {
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> src_y;
  sycl::accessor<uint8_t, 1, sycl::access::mode::read> src_uv;
  sycl::accessor<uint8_t, 1, sycl::access::mode::write> dst_y;
  sycl::accessor<uint8_t, 1, sycl::access::mode::write> dst_uv;
  int width;
  int height;
  int stride;

  DetileNV12Kernel(
      sycl::accessor<uint8_t, 1, sycl::access::mode::read> src_y,
      sycl::accessor<uint8_t, 1, sycl::access::mode::read> src_uv,
      sycl::accessor<uint8_t, 1, sycl::access::mode::write> dst_y,
      sycl::accessor<uint8_t, 1, sycl::access::mode::write> dst_uv,
      int width,
      int height,
      int stride)
      : src_y(src_y), src_uv(src_uv), dst_y(dst_y), dst_uv(dst_uv), width(width), height(height), stride(stride) {}

  // helper for TileY (Legacy Gen12)
  size_t get_offset_tiley(int x, int y, int region_stride) const {
      const int TileW = 128;
      const int TileH = 32;
      
      int tile_x = x / TileW;
      int tile_y = y / TileH;
      
      int x_in = x % TileW;
      int y_in = y % TileH;
      
      // TileY Swizzle:
      // Bit 0-3: x[0-3]
      // Bit 4-8: y[0-4]
      // Bit 9-10: x[4-6] ?? No.
      // 
      // Correct byte-level map for TileY (Column Major OWords):
      // OWord index (0-7) = x_in / 16
      // OWord offset = x_in % 16
      // 
      // Offset = (OWord_Idx * TileH + y_in) * 16 + OWord_Offset
      int oword_idx = x_in / 16;
      int offset_in_tile = (oword_idx * TileH + y_in) * 16 + (x_in % 16);
      
      // Global
      int stride_tiles = region_stride / TileW;
      return (size_t)(tile_y * stride_tiles + tile_x) * 4096 + offset_in_tile;
  }

  // helper for Tile4 (Xe-HP / Arc / PVC)
  // Tile4 is 4KB tile, but shape is 128B x 32Rows.
  // BUT internal layout is different to match 128B cachelines.
  // It is usually Row-Major of 128B lines? No.
  //
  // Tile4 Layout:
  // 128B (Cacheline) x 32 Rows.
  // Physical Memory: 
  //   Line 0: Row 0
  //   Line 1: Row 1
  //   ...
  // This is effectively LINEAR inside the tile? 
  // IF Tile4 is used, and the surface is 4KB aligned, it behaves remarkably like Linear 
  // *except* for the jumps between tiles if Stride != Width.
  // 
  // However, modern drivers often map Tile4 as "Linear" to the kernel if accessing via USM?
  // No, USM Device Allocations are linear. Imported DMABUFs are tiled.
  //
  // Let's assume Pixel-Major (Linear-like) inside the tile for Tile4.
  size_t get_offset_tile4(int x, int y, int region_stride) const {
     const int TileW = 128;
      const int TileH = 32;
      
      int tile_x = x / TileW;
      int tile_y = y / TileH;
      
      int x_in = x % TileW;
      int y_in = y % TileH;
      
      // Tile4 often maps linearly inside the 4KB page for Y-plane data types?
      // "Standard" Tile4: 
      int offset_in_tile = y_in * TileW + x_in; 
      
      int stride_tiles = region_stride / TileW;
      return (size_t)(tile_y * stride_tiles + tile_x) * 4096 + offset_in_tile;
  }

  size_t get_tiled_offset(int x, int y, int region_stride) const {
      // Toggle here based on architecture if known.
      // Gen12 (TGL/RocketLake) -> TileY
      // Gen12.7 / Xe-HP (Arc/Flex/SPR) -> Tile4
      // VAAPI Export usually provides modifier.
      //
      // Trying TileY (Column-Major) first as it is most distinct.
      // The previous implementation was: (x/16)*512 + y*16 + x%16.
      // Which IS (oword * 32 * 16) + ... wait.
      // My previous code: (x/16)*512 + (y*16).
      // 512 = 32 * 16. Correct.
      //
      // NOTE: PSNR 14dB implies we are close but misalignment exists.
      //
      // Let's try the Tile4 logic (Linear inside Tile).
      // If the image was scrambled with TileY, maybe it's Tile4.
      
      // Try TileY again but simpler verification
      // return get_offset_tiley(x, y, region_stride);
      
      // SWITCHING TO TILE4
      // Reason: Modern Intel Data Center GPU Flex/Max (Spring Hill / PVC) use Tile4 defaults.
      // Tile4 is effectively "Linear-inside-Tile" for 128B rows.
      return get_offset_tile4(x, y, region_stride);
  }

  void operator()(sycl::id<2> idx) const {
    int x = idx[1];
    int y = idx[0];

    if (x >= width || y >= height) return;

    // Detile Y Plane
    size_t linear_idx_y = y * stride + x;
    size_t tiled_idx_y = get_tiled_offset(x, y, stride);
    dst_y[linear_idx_y] = src_y[tiled_idx_y]; // Copy

    // Detile UV Plane (Half height, same stride as Y in bytes usually)
    // UV height is h/2.
    if (y < height / 2) {
       // UV plane is usually interleaved (U V U V), so width is effectively same in bytes?
       // Usually stride is bytes per row.
       // In NV12, UV stride = Y stride.
       // We process UV rows 1:1 with Y rows up to h/2.
       // X here iterates 0..width.
       // But UV width is also often handled differently in tiling.
       // Assuming standard handling:
       size_t linear_idx_uv = y * stride + x;
       size_t tiled_idx_uv = get_tiled_offset(x, y, stride); // UV plane often has same tile width logic
       dst_uv[linear_idx_uv] = src_uv[tiled_idx_uv];
    }
  }
};

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
    
    fprintf(stderr, "[DETILE_DEBUG] Starting detile kernel: width=%d, height=%d, stride=%d\n", 
            width, height, stride);
    
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
                
                /////------ XOR block for Y-Tiling (Gen12) ------///// this causes movement to up on determined columns
                ///---int x_xor_value = (x_in_tile >> 6) & 0x1;
                ///---y_in_tile ^= (x_xor_value << 2);


                // Block position added to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
                int block_x = x_in_tile / 64;  // width of pixel blocks
                int block_y = y_in_tile / 4;   // heigh of pixel blocks



                // Y-Tiling: Column-major OWord layout
                // OWord index (0-7): which 16-byte column within the tile
                int oword_idx = x_in_tile / OWordSize;
                // Offset within OWord (0-15)
                int offset_in_oword = x_in_tile % OWordSize;

                // Apply Y-Tiing XOR pattern
                ///------ int twist_y = (y_in_tile >> 2 )& 0x1;  //// This enable did not work
                ///------ oword_idx = oword_idx ^ twist_y;    //// This enable did not work

                

                
                // Offset within tile: 
                // Each OWord is TileH rows tall (32 rows * 16 bytes = 512 bytes per OWord)
                // offset = (oword_idx * TileH + y_in_tile) * OWordSize + offset_in_oword
                ////int offset_in_tile = (oword_idx * TileH + y_in_tile) * OWordSize + offset_in_oword;

                //---THIS WORKED TO A MOVEMENT OF LAGE BLOCKS, WAS THE BETTER RESULT  
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

                /// ----- int offset_in_tile = (oword_idx * TileH + y_in_tile) * OWordSize + offset_in_oword;  ////---- This enable did not worked


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
    
    fprintf(stderr, "[DETILE_DEBUG] Detile kernel completed\n");
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