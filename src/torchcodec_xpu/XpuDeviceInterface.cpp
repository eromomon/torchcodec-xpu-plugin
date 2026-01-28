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
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

  const bool USE_SYCL_COLOR_CONVERSION_KERNEL = true;

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

  if (USE_SYCL_COLOR_CONVERSION_KERNEL) {
    std::cout << "XpuDeviceInterface initialized with SYCL kernel backend" << std::endl;
    VLOG(1) << "Backend: SYCL_KERNEL (Direct NV12→RGB)";
  } else {
    std::cout << "XpuDeviceInterface initialized with VAAPI filter graph backend" << std::endl;
    VLOG(1) << "Backend: VAAPI_FILTER (Flexible, with scaling)";
  }
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
  // Relaxed validations to support multi-layer descriptors (e.g. separate Y/UV planes description)
  // TORCH_CHECK(desc.num_layers == 1, "Expected 1 layer, got ", desc.num_layers);
  // TORCH_CHECK(
  //     desc.layers[0].num_planes == 1,
  //     "Expected 1 plane, got ",
  //     desc.layers[0].num_planes);

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

  // Check for Tiling (Intel Gen12+ uses Y-tiling which requires de-tiling for linear access)
  // When vaExportSurfaceHandle returns DRM_PRIME_2, we get modifiers.
  bool is_tiled = (desc.objects[0].drm_format_modifier != 0); // Non-linear
  void* usm_ptr = nullptr;
  size_t alloc_size = 0;
  bool is_rgb = false;

  if (is_tiled) {
    printf(">>>>> TILED\n");
    // Tiled Memory path: Map (De-tile) -> Upload NV12 -> SYCL Kernel (Bilinear)
    // Moving workload to GPU while maintaining high quality via improved kernel.
    
    VADisplay vaDisplay = getVaDisplayFromAV(frame.get());
    VASurfaceID surfaceID = (VASurfaceID)(uintptr_t)frame->data[3];
    VAImage image;

# if 0
    // 1. Create a VAImage to perform the De-Tiling copy on driver side
    VAImageFormat val_fmt = {VA_FOURCC_NV12, VA_LSB_FIRST, 12, 0, 0, 0, 0, 0, {}};
    sts = vaCreateImage(vaDisplay, &val_fmt, desc.width, desc.height, &image);
    TORCH_CHECK(sts == VA_STATUS_SUCCESS, "vaCreateImage failed: ", vaErrorStr(sts));

    // 2. Derive Image (Copy Surface -> Image)
    sts = vaGetImage(vaDisplay, surfaceID, 0, 0, desc.width, desc.height, image.image_id);
    TORCH_CHECK(sts == VA_STATUS_SUCCESS, "vaGetImage failed: ", vaErrorStr(sts));
#endif

    sts = vaDeriveImage(vaDisplay, surfaceID, &image);
    TORCH_CHECK(sts == VA_STATUS_SUCCESS, "vaDeriveImage failed: ", vaErrorStr(sts));

    // 3. Map Buffer to Host (Result is Linear NV12 in Host Memory)
    void* host_addr = nullptr;
    sts = vaMapBuffer(vaDisplay, image.buf, &host_addr);
    TORCH_CHECK(sts == VA_STATUS_SUCCESS, "vaMapBuffer failed: ", vaErrorStr(sts));

    printf(">>> image.format.fourcc=%x\n", image.format.fourcc);
    printf(">>> image.num_planes=%d\n", image.num_planes);
    printf(">>> image.width=%d\n", image.width);
    printf(">>> image.height=%d\n", image.height);
    printf(">>> image.pitches[0]=%d\n", image.pitches[0]);
    printf(">>> image.pitches[1]=%d\n", image.pitches[1]);
    printf(">>> image.offsets[0]=%d\n", image.offsets[0]);
    printf(">>> image.offsets[1]=%d\n", image.offsets[1]);


    int w = frame->width;
    int h = frame->height;
    int stride = image.pitches[0]; // Assuming Y/UV stride equal

    // Determine Colorspace (BT.709 vs BT.601)
    int colorspace = frame->colorspace;
    int color_std = 1; // Default BT.709
    
    if (colorspace == AVCOL_SPC_UNSPECIFIED) {
        if (w >= 1280 || h >= 720) colorspace = AVCOL_SPC_BT709;
        else colorspace = AVCOL_SPC_SMPTE170M;
    }
    if (colorspace == AVCOL_SPC_SMPTE170M || colorspace == AVCOL_SPC_BT470BG) {
        color_std = 0; // BT.601
    }

    // 4. Allocate Device Memory (Output RGB)
    alloc_size = w * h * 3;
    ze_device_mem_alloc_desc_t alloc_desc_usm = {};
    alloc_desc_usm.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc_usm.ordinal = 0;
    
    ze_result_t res = zeMemAllocDevice(context->zeCtx, &alloc_desc_usm, alloc_size, 0, ze_device, &usm_ptr);
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (RGB) failed");

    // 5. Allocate Temporary Device Memory for NV12 Input (Y + UV)
    void* dev_y = nullptr;
    void* dev_uv = nullptr;
    size_t y_size = stride * h;
    size_t uv_size = stride * h / 2;

    res = zeMemAllocDevice(context->zeCtx, &alloc_desc_usm, y_size, 0, ze_device, &dev_y);
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (Y) failed");
    
    res = zeMemAllocDevice(context->zeCtx, &alloc_desc_usm, uv_size, 0, ze_device, &dev_uv);
    TORCH_CHECK(res == ZE_RESULT_SUCCESS, "zeMemAllocDevice (UV) failed");

    // 6. Upload NV12 from Host to Device
    uint8_t* host_y = (uint8_t*)host_addr + image.offsets[0];
    uint8_t* host_uv = (uint8_t*)host_addr + image.offsets[1]; // UV is usually at offset[1]

    // Use async copies then wait
    queue.memcpy(dev_y, host_y, y_size);
    queue.memcpy(dev_uv, host_uv, uv_size).wait();

    // 7. Run High-Quality SYCL Kernel (Bilinear)
    convertNV12ToRGB(queue, (uint8_t*)dev_y, (uint8_t*)dev_uv, (uint8_t*)usm_ptr, w, h, stride, !color_std);

    // 8. Cleanup Temporary Device Memory
    zeMemFree(context->zeCtx, dev_y);
    zeMemFree(context->zeCtx, dev_uv);

    // 9. Cleanup VAAPI Image
    vaUnmapBuffer(vaDisplay, image.buf);
    vaDestroyImage(vaDisplay, image.image_id);
    
    // Clean up original FD import attempt
    close(desc.objects[0].fd);
    is_rgb = true;

  } else {
    printf(">>>>> LINEAR\n");
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

    close(desc.objects[0].fd);
  }

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
  // Safe copy of shape array for valid pointer lifetime if needed, 
  // but DLPack struct stores pointer. We must ensure the array persists? 
  // Wait, standard DLManagedTensor doesn't own the shape array? 
  // Usually shape is allocated continuously. 
  // The original code used `int64_t shape[3]` on stack! That is a BUG in the original code unless 
  // `dl_dst->dl_tensor.shape` is copied immediately by at::fromDLPack. 
  // Torch DOES copy it. So stack is fine? 
  // But purely, `dl_tensor.shape` is `int64_t*`. If `fromDLPack` doesn't copy immediately, this explodes.
  // The original code:
  // int64_t shape[3] = ...
  // dl_dst->dl_tensor.shape = shape;
  // auto dst = at::fromDLPack(...) 
  // Valid because `dst` creation happens before function return.

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

void XpuDeviceInterface::convertAVFrameToFrameOutput_SYCL(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  // TODO: consider to copy handling of CPU frame from CUDA
  // TODO: consider to copy NV12 format check from CUDA
  //
  printf(">>>>>>>>>>>>> HEELLOO\n");
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


  // Distinguish two paths: SYCL Kernel implementation vs VAAPI Filter Graph implementation
//  if (USE_SYCL_KERNEL_BACKEND) {
//    // Path 1: SYCL Kernel Backend
//    VLOG(2) << "Using SYCL kernel backend for conversion";
//    convertAVFrameToFrameOutput_SYCL(avFrame, frameOutput, frameDims, dst);
//  } else {
//    // Path 2: VAAPI Filter Graph Backend
//    VLOG(2) << "Using VAAPI filter graph backend for conversion";
//    convertAVFrameToFrameOutput_FilterGraph(avFrame, frameOutput, frameDims, dst);
//  }

  // Check if we can do a direct/high-quality conversion via AVFrameToTensor (e.g. Tiled sws_scale path)
  // This bypasses the VAAPI filter graph if the export handles color conversion.
  torch::Tensor direct_tensor = AVFrameToTensor(device_, avFrame);
  if (direct_tensor.size(2) == 3) {
      // We got RGB directly (sws_scale path)
      dst.copy_(direct_tensor);
      return;
  }
  // If not RGB (e.g. Linear NV12), fall back to standard VAAPI Filter Graph conversion
  // UPDATE: User requested to replace the generic VAAPI filter graph with the custom SYCL ColorConversionKernel.
  
  // 1. Get Surface Descriptor to find strides and offsets for NV12
  VADRMPRIMESurfaceDescriptor desc{};
  VAStatus sts = vaExportSurfaceHandle(
      getVaDisplayFromAV(avFrame.get()),
      (VASurfaceID)(uintptr_t)avFrame->data[3],
      VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
      VA_EXPORT_SURFACE_READ_ONLY,
      &desc);
  TORCH_CHECK(sts == VA_STATUS_SUCCESS, "vaExportSurfaceHandle failed");
  
  // Close FD immediately as we only need layout info, and direct_tensor holds the memory reference
  if (desc.num_objects > 0) {
      close(desc.objects[0].fd);
  }

  // 2. Calculate Offsets
  // Assuming NV12 Linear in one object (standard for mappable GEM/USM)
  // If independent layers, logical arithmetic applies.
  uint32_t stride = desc.layers[0].pitch[0];
  uint32_t y_offset = desc.layers[0].offset[0];
  uint32_t uv_offset = 0;
  
  if (desc.num_layers > 1) {
      uv_offset = desc.layers[1].offset[0];
  } else if (desc.layers[0].num_planes > 1) {
      uv_offset = desc.layers[0].offset[1]; 
  } else {
      uv_offset = y_offset + stride * avFrame->height;
  }

  // 3. Get Pointers
  // direct_tensor.data_ptr() points to the base of the USM allocation offset by dl_tensor.byte_offset.
  // In our AVFrameToTensor implementation: dl_tensor.byte_offset = desc.layers[0].offset[0];
  // So data_ptr() is effectively &USM_Base[y_offset].
  uint8_t* y_ptr = (uint8_t*)direct_tensor.data_ptr();
  
  // UV is at some offset relative to Y
  // uv_ptr = &USM_Base[uv_offset]
  //        = &USM_Base[y_offset] + (uv_offset - y_offset)
  long relative_uv_offset = (long)uv_offset - (long)y_offset;
  uint8_t* uv_ptr = y_ptr + relative_uv_offset;

  // 4. Run Kernel
  sycl::queue queue = c10::xpu::getCurrentXPUStream(device_.index());
  
  // Determine Colorspace (BT.709 vs BT.601)
  int colorspace = avFrame->colorspace;
  int color_std = 1; // Default BT.709
  
  if (colorspace == AVCOL_SPC_UNSPECIFIED) {
      if (frameDims.width >= 1280 || frameDims.height >= 720) colorspace = AVCOL_SPC_BT709;
      else colorspace = AVCOL_SPC_SMPTE170M;
  }
  if (colorspace == AVCOL_SPC_SMPTE170M || colorspace == AVCOL_SPC_BT470BG) {
      color_std = 0; // BT.601
  }
  
  auto start = std::chrono::high_resolution_clock::now();

  /*double rgb2yuv[3][3], yuv2rgb[3];
  double tmp_mat[3][3];
  ff_fill_rgb2yuv_table(s->in_lumacoef, rgb2yuv);
  ff_matrix_invert_3x3(rgb2yuv, &yuv2rgb);*/

  convertNV12ToRGB(
      queue,
      y_ptr,
      uv_ptr,
      (uint8_t*)dst.data_ptr(),
      frameDims.width,
      frameDims.height,
      stride,
      !color_std
  );

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "Conversion (SYCL) of frame height=" << frameDims.height << " width=" << frameDims.width
          << " took: " << duration.count() << "us" << std::endl;
}

void XpuDeviceInterface::convertAVFrameToFrameOutput_FilterGraph(
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

  auto start = std::chrono::high_resolution_clock::now();
  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);
  FiltersContext filtersContext;

  filtersContext.inputWidth = avFrame->width;
  filtersContext.inputHeight = avFrame->height;
  filtersContext.inputFormat = frameFormat;
  filtersContext.inputAspectRatio = avFrame->sample_aspect_ratio;
  // Actual output color format will be set via filter options
  filtersContext.outputFormat = AV_PIX_FMT_VAAPI;
  filtersContext.timeBase = timeBase_;
  filtersContext.hwFramesCtx.reset(av_buffer_ref(avFrame->hw_frames_ctx));

  std::stringstream filters;
  filters << "scale_vaapi=" << frameDims.width << ":" << frameDims.height;
  // CPU device interface outputs RGB in full (pc) color range.
  // We are doing the same to match.
  filters << ":format=rgba:out_range=pc";

  filtersContext.filtergraphStr = filters.str();

  if (!filterGraphContext_ || prevFiltersContext_ != filtersContext) {
    filterGraphContext_ =
        std::make_unique<FilterGraph>(filtersContext, videoStreamOptions_);
    prevFiltersContext_ = std::move(filtersContext);
  }

  // We convert input to the RGBX color format with VAAPI getting WxHx4
  // tensor on the output.
  UniqueAVFrame filteredAVFrame = filterGraphContext_->convert(avFrame);

  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_VAAPI);

  torch::Tensor dst_rgb4 = AVFrameToTensor(device_, filteredAVFrame);
  dst.copy_(dst_rgb4.narrow(2, 0, 3));

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::micro> duration = end - start;
  VLOG(9) << "Conversion of frame height=" << frameDims.height << " width=" << frameDims.width
          << " took: " << duration.count() << "us" << std::endl;
}


void XpuDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  
  // ===== VALIDATION =====
  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_VAAPI,
      "Expected VAAPI format");
  
  auto frameDims = FrameDims(avFrame->height, avFrame->width);
  torch::Tensor& dst = frameOutput.data;
  
  // ===== ALLOCATE OUTPUT =====
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(frameDims, device_);
  }
  
  // ===== DISPATCHER: HARDCODED SELECTION =====
  //
  // The key decision point: which backend to use?
  // This is evaluated at compile-time (zero overhead)
  //
  if (USE_SYCL_COLOR_CONVERSION_KERNEL) {
    // ===== BRANCH A: SYCL Kernel =====
    // Fast, direct NV12→RGB conversion
    // Limited to NV12 format
    // High performance for tiled/linear surfaces
    convertAVFrameToFrameOutput_SYCL(avFrame, frameOutput, preAllocatedOutputTensor);
  } else {
    // ===== BRANCH B: VAAPI Filter Graph =====
    // Flexible, handles multiple formats
    // Supports scaling via scale_vaapi filter
    // Converts via RGBA intermediate (4 channels)
    convertAVFrameToFrameOutput_FilterGraph(avFrame, frameOutput, preAllocatedOutputTensor);
  }
  
  // ===== POST-PROCESSING (Common to Both) =====
  // (Any common cleanup or validation here)
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
