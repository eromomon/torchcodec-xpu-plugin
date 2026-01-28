// Copyright (c) 2025 Dmitry Rogozhkin.

#pragma once

#include "DeviceInterface.h"
#include "FilterGraph.h"

namespace facebook::torchcodec {

class XpuDeviceInterface : public DeviceInterface {
 public:
  XpuDeviceInterface(const torch::Device& device);

  virtual ~XpuDeviceInterface();

  std::optional<const AVCodec*> findCodec(
      const AVCodecID& codecId,
      bool isDecoder = true) override;

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void initializeVideo(
      const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      [[maybe_unused]] const std::optional<FrameDims>& resizedOutputDims)
      override;

  void registerHardwareDeviceWithCodec(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;

  UniqueAVBufferRef ctx_;

  std::unique_ptr<FilterGraph> filterGraphContext_;
  FiltersContext prevFiltersContext_;

  void convertAVFrameToFrameOutput_SYCL(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor);

  void convertAVFrameToFrameOutput_FilterGraph(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor);


};

} // namespace facebook::torchcodec
