// Copyright (c) 2025 Dmitry Rogozhkin.

#include <unistd.h>

#include <level_zero/ze_api.h>
#include <va/va_drmcommon.h>

#include <ATen/DLConvertor.h>
#include <c10/xpu/XPUStream.h>

#include "ColorConversionKernel.h"

extern "C" {
#include <libswscale/swscale.h>
}

#include "Cache.h"
#include "FFMPEGCommon.h"
#include "XpuDeviceInterface.h"

extern "C" {
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {
namespace {

static bool g_xpu = registerDeviceInterface(
    DeviceInterfaceKey(torch::kXPU),
    [](const torch::Device& device) { return new XpuDeviceInterface(device); });

const int MAX_XPU_GPUS = 128;
// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;
PerGpuCache<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>
    g_cached_hw_device_ctxs(MAX_XPU_GPUS, MAX_CONTEXTS_PER_GPU_IN_CACHE);

UniqueAVBufferRef getVaapiContext(const torch::Device& device) {
  enum AVHWDeviceType type = av_hwdevice_find_type_by_name("vaapi");
  TORCH_CHECK(type != AV_HWDEVICE_TYPE_NONE, "Failed to find vaapi device");
  int deviceIndex = getDeviceIndex(device);

  UniqueAVBufferRef hw_device_ctx = g_cached_hw_device_ctxs.get(device);
  if (hw_device_ctx) {
    return hw_device_ctx;
  }

  std::string renderD = "/dev/dri/renderD128";

  sycl::device syclDevice = c10::xpu::get_raw_device(deviceIndex);
  if (syclDevice.has(sycl::aspect::ext_intel_pci_address)) {
    auto BDF =
        syclDevice.get_info<sycl::ext::intel::info::device::pci_address>();
    renderD = "/dev/dri/by-path/pci-" + BDF + "-render";
  }

  AVBufferRef* ctx = nullptr;
  int err = av_hwdevice_ctx_create(&ctx, type, renderD.c_str(), nullptr, 0);
  if (err < 0) {
    TORCH_CHECK(
        false,
        "Failed to create specified HW device: ",
        getFFMPEGErrorStringFromErrorCode(err));
  }
  return UniqueAVBufferRef(ctx);
}

} // namespace

int getDeviceIndex(const torch::Device& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and XPU
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_XPU_GPUS,
      "Invalid device index = ",
      deviceIndex);

  return (deviceIndex == -1)? 0: deviceIndex;
}

XpuDeviceInterface::XpuDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_xpu, "XpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kXPU, "Unsupported device: ", device_.str());

  // It is important for pytorch itself to create the xpu context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the xpu context.
  torch::Tensor dummyTensorForXpuInitialization = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
  ctx_ = getVaapiContext(device_);
}

XpuDeviceInterface::~XpuDeviceInterface() {
  if (ctx_) {
    g_cached_hw_device_ctxs.addIfCacheHasCapacity(device_, std::move(ctx_));
  }
}

void XpuDeviceInterface::initialize(
    const AVStream* avStream,
    [[maybe_unused]] const UniqueDecodingAVFormatContext& avFormatCtx,
    [[maybe_unused]] const SharedAVCodecContext& codecContext) {
  TORCH_CHECK(avStream != nullptr, "avStream is null");
  codecContext_ = codecContext;
  timeBase_ = avStream->time_base;
}

void XpuDeviceInterface::initializeVideo(
    const VideoStreamOptions& videoStreamOptions,
    [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>& transforms,
    [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims) {
  videoStreamOptions_ = videoStreamOptions;
}

void XpuDeviceInterface::registerHardwareDeviceWithCodec(
    AVCodecContext* codecContext) {
  TORCH_CHECK(ctx_, "FFmpeg HW device has not been initialized");
  TORCH_CHECK(codecContext != nullptr, "codecContext is null");
  codecContext->hw_device_ctx = av_buffer_ref(ctx_.get());
}

VADisplay getVaDisplayFromAV(AVFrame* avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

struct xpuManagerCtx {
  UniqueAVFrame avFrame;
  ze_context_handle_t zeCtx = nullptr;
};

void deleter(DLManagedTensor* self) {
  std::unique_ptr<DLManagedTensor> tensor(self);
  std::unique_ptr<xpuManagerCtx> context((xpuManagerCtx*)self->manager_ctx);
  zeMemFree(context->zeCtx, self->dl_tensor.data);
}

torch::Tensor AVFrameToTensor(
    const torch::Device& device,
    const UniqueAVFrame& frame) {
  TORCH_CHECK_EQ(frame->format, AV_PIX_FMT_VAAPI);

  VADRMPRIMESurfaceDescriptor desc{};

  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(frame.get()),
      (VASurfaceID)(uintptr_t)frame->data[3],
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(
      sts == VA_STATUS_SUCCESS,
      "vaExportSurfaceHandle failed: ",
      vaErrorStr(sts));

  TORCH_CHECK(desc.num_objects == 1, "Expected 1 fd, got ", desc.num_objects);

  std::unique_ptr<xpuManagerCtx> context = std::make_unique<xpuManagerCtx>();
  ze_device_handle_t ze_device{};
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device.index());

  queue
      .submit([&](sycl::handler& cgh) {
        cgh.host_task([&](const sycl::interop_handle& ih) {
          context->zeCtx =
              ih.get_native_context<sycl::backend::ext_oneapi_level_zero>();
          ze_device =
              ih.get_native_device<sycl::backend::ext_oneapi_level_zero>();
        });
      })
      .wait();


  bool is_tiled = (desc.objects[0].drm_format_modifier != 0); // Non-linear
  void* usm_ptr = nullptr;
  size_t alloc_size = 0;
  bool is_rgb = false;

  // Import Memory (used for both Tiled and Linear source)
  // ... refactored to specific paths below ...
  
  if (is_tiled) {
    // ========== OPTION C: Pure SYCL Path ==========
    // Tiled Surface (DRM_PRIME_2) → Custom SYCL Detile Kernel → Color Conversion
    // Zero-copy approach: everything stays on device

    // 1. Import tiled surface directly to device memory via DMA-BUF FD
    ze_external_memory_import_fd_t import_fd_desc{};
    import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    import_fd_desc.fd = desc.objects[0].fd;
    
    ze_device_mem_alloc_desc_t alloc_dev = {};
    alloc_dev.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_dev.pNext = &import_fd_desc;  // ← Critical: chain the import descriptor
    alloc_dev.ordinal = 0;
    
    void* tiled_device_ptr = nullptr;
    ze_result_t res = zeMemAllocDevice(
        context->zeCtx,
        &alloc_dev,
        desc.objects[0].size,
        0,
        ze_device,
        &tiled_device_ptr
    );
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (tiled import) failed");
        
    // 2. Calculate linear buffer sizes based on physical stride
    uint32_t physical_stride = desc.layers[0].pitch[0];
    size_t y_linear_size = physical_stride * desc.height;
    size_t uv_linear_size = physical_stride * (desc.height / 2);
        
    // 3. Allocate linear output buffers on device
    void* linear_y_ptr = nullptr;
    void* linear_uv_ptr = nullptr;
    
    // Create fresh descriptor WITHOUT import chain for linear allocations
    ze_device_mem_alloc_desc_t alloc_linear = {};
    alloc_linear.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_linear.pNext = nullptr;  // ← Critical: no import for fresh allocations
    alloc_linear.ordinal = 0;
    
    res = zeMemAllocDevice(context->zeCtx, &alloc_linear, y_linear_size, 0, ze_device, &linear_y_ptr);
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (linear Y) failed");
    
    res = zeMemAllocDevice(context->zeCtx, &alloc_linear, uv_linear_size, 0, ze_device, &linear_uv_ptr);
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (linear UV) failed");
    
    // 4. Call custom detiling kernel (SYCL)
    // Detile Y and UV planes from tiled to linear format
    uint8_t* tiled_y_ptr = (uint8_t*)tiled_device_ptr + desc.layers[0].offset[0];
    uint8_t* tiled_uv_ptr = nullptr;
    
    // Handle both single-layer and multi-layer UV descriptors
    if (desc.num_layers > 1) {
      // Multi-layer: separate Y and UV layers
      tiled_uv_ptr = (uint8_t*)tiled_device_ptr + desc.layers[1].offset[0];
    } else if (desc.layers[0].num_planes >= 2) {
      // Single layer with multiple planes: use plane offset
      tiled_uv_ptr = (uint8_t*)tiled_device_ptr + desc.layers[0].offset[1];
    } else {
      TORCH_CHECK(false, "Cannot determine UV plane offset: layers=", desc.num_layers, 
                  ", planes=", desc.layers[0].num_planes);
    }
    
    // Validate pointers are within allocated buffer
    size_t y_offset_from_base = (size_t)tiled_y_ptr - (size_t)tiled_device_ptr;
    size_t uv_offset_from_base = (size_t)tiled_uv_ptr - (size_t)tiled_device_ptr;
    
    TORCH_CHECK(y_offset_from_base < desc.objects[0].size, 
                "Y plane offset ", y_offset_from_base, " exceeds buffer size ", desc.objects[0].size);
    TORCH_CHECK(uv_offset_from_base < desc.objects[0].size,
                "UV plane offset ", uv_offset_from_base, " exceeds buffer size ", desc.objects[0].size);
    TORCH_CHECK(linear_y_ptr != nullptr && linear_uv_ptr != nullptr,
                "Linear buffers not allocated");
    
    try {
      detachTiledNV12(
          tiled_y_ptr,
          tiled_uv_ptr,
          (uint8_t*)linear_y_ptr,
          (uint8_t*)linear_uv_ptr,
          desc.width,
          desc.height,
          physical_stride,
          queue
      );
    } catch (const sycl::exception& e) {
      TORCH_CHECK(false, "SYCL exception in detiling kernel: ", e.what());
    } catch (const std::exception& e) {
      throw;
    }
    // Debug: Sample first pixel of Y and UV to verify data
    uint8_t sample_y[1], sample_uv[2];
    queue.memcpy(sample_y, linear_y_ptr, 1).wait();
    queue.memcpy(sample_uv, linear_uv_ptr, 2).wait();
    
    
    // 5. Allocate RGB output buffer on device
    size_t rgb_size = frame->width * frame->height * 3;
    
    //*-fprintf(stderr, "[DEBUG] About to allocate RGB buffer: %lu bytes\n", (unsigned long)rgb_size);
    //*-fflush(stderr);
    
    // Use separate descriptor for RGB allocation (not the import descriptor)
    ze_device_mem_alloc_desc_t alloc_rgb = {};
    alloc_rgb.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_rgb.pNext = nullptr;  // No import
    alloc_rgb.ordinal = 0;
    
    res = zeMemAllocDevice(context->zeCtx, &alloc_rgb, rgb_size, 0, ze_device, &usm_ptr);
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (RGB) failed");
    
    // 6. Determine colorspace
    int colorspace = frame->colorspace;
    int color_std = 1;  // BT.709 (HD) - default
    
    if (colorspace == AVCOL_SPC_UNSPECIFIED) {
        // Heuristic: use resolution to guess colorspace if not specified
        if (frame->width >= 1280 || frame->height >= 720) {
            colorspace = AVCOL_SPC_BT709;
        } else {
            colorspace = AVCOL_SPC_SMPTE170M;
        }
    }
    
    if (colorspace == AVCOL_SPC_SMPTE170M || colorspace == AVCOL_SPC_BT470BG) {
        color_std = 0;  // BT.601 (SD)
    }
    
    // 7. Run color conversion on linear data
    auto conversion_start = std::chrono::high_resolution_clock::now();
    
    try {
      convertNV12ToRGB(
          (uint8_t*)linear_y_ptr,
          (uint8_t*)linear_uv_ptr,
            (uint8_t*)usm_ptr,
          frame->width,
          frame->height,
          physical_stride,
          queue,
          color_std
      );
      
    } catch (const sycl::exception& e) {
      TORCH_CHECK(false, "SYCL exception in color conversion: ", e.what());
    } catch (const std::exception& e) {
      throw;
    }
    
    auto conversion_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> conversion_duration = conversion_end - conversion_start;

    // 8. Cleanup intermediate buffers (keep usm_ptr for output)
    zeMemFree(context->zeCtx, tiled_device_ptr);
    zeMemFree(context->zeCtx, linear_y_ptr);
    zeMemFree(context->zeCtx, linear_uv_ptr);
    
    is_rgb = true;

  } else {
    // Linear path (Zero Copy)
    ze_external_memory_import_fd_t import_fd_desc{};
    import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
    import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    import_fd_desc.fd = desc.objects[0].fd;

    ze_device_mem_alloc_desc_t alloc_desc{};
    alloc_desc.pNext = &import_fd_desc;
    
    alloc_size = desc.objects[0].size;

    ze_result_t res = zeMemAllocDevice(
        context->zeCtx,
        &alloc_desc,
        alloc_size,
        0,
        ze_device,
        &usm_ptr);
    TORCH_CHECK(
        res == ZE_RESULT_SUCCESS, "Failed to import fd=", desc.objects[0].fd);
  }

  // Close FD since we imported it
  close(desc.objects[0].fd);

  std::unique_ptr<DLManagedTensor> dl_dst = std::make_unique<DLManagedTensor>();
  // Update shape based on IS_RGB
  int64_t shape[3];
  if (is_rgb) {
      shape[0] = frame->height;
      shape[1] = frame->width;
      shape[2] = 3; // RGB
  } else {
      shape[0] = desc.height;
      shape[1] = desc.width;
      shape[2] = 4; // Native (likely to be reinterpreted)
  }

  context->avFrame.reset(av_frame_alloc());
  TORCH_CHECK(context->avFrame.get(), "Failed to allocate AVFrame");

  int status = av_frame_ref(context->avFrame.get(), frame.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to reference AVFrame: ",
      getFFMPEGErrorStringFromErrorCode(status));

  dl_dst->manager_ctx = context.release();
  dl_dst->deleter = deleter;
  dl_dst->dl_tensor.data = usm_ptr;
  dl_dst->dl_tensor.device.device_type = kDLOneAPI;
  dl_dst->dl_tensor.device.device_id = device.index();
  dl_dst->dl_tensor.ndim = 3;
  dl_dst->dl_tensor.dtype.code = kDLUInt;
  dl_dst->dl_tensor.dtype.bits = 8;
  dl_dst->dl_tensor.dtype.lanes = 1;
  dl_dst->dl_tensor.shape = shape;
  dl_dst->dl_tensor.strides = nullptr;
  dl_dst->dl_tensor.byte_offset = desc.layers[0].offset[0];

  auto dst = at::fromDLPack(dl_dst.release());

  return dst;
}

VADisplay getVaDisplayFromAV(UniqueAVFrame& avFrame) {
  AVHWFramesContext* hwfc = (AVHWFramesContext*)avFrame->hw_frames_ctx->data;
  AVHWDeviceContext* hwdc = hwfc->device_ctx;
  AVVAAPIDeviceContext* vactx = (AVVAAPIDeviceContext*)hwdc->hwctx;
  return vactx->display;
}

void XpuDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // TODO: consider to copy handling of CPU frame from CUDA
  // TODO: consider to copy NV12 format check from CUDA
  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_VAAPI,
      "Expected format to be AV_PIX_FMT_VAAPI, got " +
          std::string(av_get_pix_fmt_name((AVPixelFormat)avFrame->format)));
  auto frameDims = FrameDims(avFrame->height, avFrame->width);
  torch::Tensor& dst = frameOutput.data;
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
	    (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        shape);
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(frameDims, device_);
  }

  // Check if we can do a direct/high-quality conversion via AVFrameToTensor (e.g. Tiled sws_scale path)
  // This bypasses the VAAPI filter graph if the export handles color conversion.
  torch::Tensor direct_tensor = AVFrameToTensor(device_, avFrame);
  if (direct_tensor.size(2) == 3) {
      // We got RGB directly (sws_scale path)
      dst.copy_(direct_tensor);
      return;
  }
}

// inspired by https://github.com/FFmpeg/FFmpeg/commit/ad67ea9
// we have to do this because of an FFmpeg bug where hardware decoding is not
// appropriately set, so we just go off and find the matching codec for the CUDA
// device
std::optional<const AVCodec*> XpuDeviceInterface::findCodec(
    const AVCodecID& codecId,
    bool isDecoder) {
  void* i = nullptr;
  const AVCodec* codec = nullptr;
  while ((codec = av_codec_iterate(&i)) != nullptr) {
    if (isDecoder) {
      if (codec->id != codecId || !av_codec_is_decoder(codec)) {
        continue;
      }
    } else {
      if (codec->id != codecId || !av_codec_is_encoder(codec)) {
        continue;
      }
    }
    

    const AVCodecHWConfig* config = nullptr;
    for (int j = 0; (config = avcodec_get_hw_config(codec, j)) != nullptr;
         ++j) {
      if (config->device_type == AV_HWDEVICE_TYPE_VAAPI) {
        return codec;
      }
    }
  }

  return std::nullopt;
}

} // namespace facebook::torchcodec
