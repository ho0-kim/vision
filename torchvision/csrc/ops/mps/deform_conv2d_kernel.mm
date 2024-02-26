/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

#include <ATen/ATen.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>
#include "mps_helpers.h"
#include "mps_kernels.h"

namespace vision {
namespace ops {

namespace {

const int kMaxParallelImgs = 32;

void deformable_im2col(
    const at::Tensor& input,
    const at::Tensor& data_offset,
    const at::Tensor& data_mask,
    int n_in_channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int parallel_imgs,
    int deformable_group,
    bool use_mask,
    at::Tensor data_col) {
  using namespace at::native::mps;

  const int64_t num_kernels =
      (int64_t)n_in_channels * out_h * out_w * parallel_imgs;

  auto input_ = input.contiguous();
  auto data_offset_ = data_offset.contiguous();
  auto data_mask_ = data_mask.contiguous();

  int64_t n_in_channels_ = (int64_t)n_in_channels;
  int64_t height_ = (int64_t)height;
  int64_t width_ = (int64_t)width;
  int64_t weight_h_ = (int64_t)weight_h;
  int64_t weight_w_ = (int64_t)weight_w;
  int64_t pad_h_ = (int64_t)pad_h;
  int64_t pad_w_ = (int64_t)pad_w;
  int64_t stride_h_ = (int64_t)stride_h;
  int64_t stride_w_ = (int64_t)stride_w;
  int64_t dilation_h_ = (int64_t)dilation_h;
  int64_t dilation_w_ = (int64_t)dilation_w;
  int64_t out_h_ = (int64_t)out_h;
  int64_t out_w_ = (int64_t)out_w;
  int64_t parallel_imgs_ = (int64_t)parallel_imgs;
  int64_t deformable_group_ = (int64_t)deformable_group;

  id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_);
  id<MTLBuffer> dataOffsetBuffer = getMTLBufferStorage(data_offset_);
  id<MTLBuffer> dataMaskBuffer = getMTLBufferStorage(data_mask_);
  id<MTLBuffer> dataColBuffer = getMTLBufferStorage(data_col);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool{
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "deformable_im2col_" + scalarToMetalTypeString(input.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, 
          /*Kernel Name*/ kernel, 
          /*Tensorlist*/{input_, data_offset_, data_mask_});

      [computeEncoder setComputePipelineState:visionPSO];

      [computeEncoder setBytes:&num_kernels length:sizeof(int64_t) atIndex:0];
      [computeEncoder setBuffer:inputBuffer offset:input_.storage_offset() * input_.element_size() atIndex:1];
      [computeEncoder setBuffer:dataOffsetBuffer offset:data_offset_.storage_offset() * data_offset_.element_size() atIndex:2];
      [computeEncoder setBuffer:dataMaskBuffer offset:data_mask_.storage_offset() * data_mask_.element_size() atIndex:3];
      [computeEncoder setBytes:&height_ length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:&width_ length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&weight_h_ length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&weight_w_ length:sizeof(int64_t) atIndex:7];
      [computeEncoder setBytes:&pad_h_ length:sizeof(int64_t) atIndex:8];
      [computeEncoder setBytes:&pad_w_ length:sizeof(int64_t) atIndex:9];
      [computeEncoder setBytes:&stride_h_ length:sizeof(int64_t) atIndex:10];
      [computeEncoder setBytes:&stride_w_ length:sizeof(int64_t) atIndex:11];
      [computeEncoder setBytes:&dilation_h_ length:sizeof(int64_t) atIndex:12];
      [computeEncoder setBytes:&dilation_w_ length:sizeof(int64_t) atIndex:13];
      [computeEncoder setBytes:&parallel_imgs_ length:sizeof(int64_t) atIndex:14];
      [computeEncoder setBytes:&n_in_channels_ length:sizeof(int64_t) atIndex:15];
      [computeEncoder setBytes:&deformable_group_ length:sizeof(int64_t) atIndex:16];
      [computeEncoder setBytes:&out_h_ length:sizeof(int64_t) atIndex:17];
      [computeEncoder setBytes:&out_w_ length:sizeof(int64_t) atIndex:18];
      [computeEncoder setBytes:&use_mask length:sizeof(bool) atIndex:19];
      [computeEncoder setBuffer:dataColBuffer offset:data_col.storage_offset() * data_col.element_size() atIndex:20];

      NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > threadsPerBlock) {
        tgSize = threadsPerBlock;
      }
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);

      NSUInteger gridSize = static_cast<NSUInteger>((num_kernels + tgSize - 1) / tgSize);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSize, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });
}

int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

void compute_grad_input(
    const at::Tensor& columns,
    const at::Tensor& offset,
    const at::Tensor& mask,
    int channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int n_offset_grps,
    bool use_mask,
    at::Tensor grad_im) {
  using namespace at::native::mps;

  const int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  const int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;

  const int64_t num_kernels =
      (int64_t)channels * weight_h * weight_w * out_h * out_w * parallel_imgs;

  auto columns_ = columns.contiguous();
  auto offset_ = offset.contiguous();
  auto mask_ = mask.contiguous();

  int64_t channels_ = (int64_t)channels;
  int64_t height_ = (int64_t)height;
  int64_t width_ = (int64_t)width;
  int64_t weight_h_ = (int64_t)weight_h;
  int64_t weight_w_ = (int64_t)weight_w;
  int64_t pad_h_ = (int64_t)pad_h;
  int64_t pad_w_ = (int64_t)pad_w;
  int64_t stride_h_ = (int64_t)stride_h;
  int64_t stride_w_ = (int64_t)stride_w;
  int64_t dilation_h_ = (int64_t)dilation_h;
  int64_t dilation_w_ = (int64_t)dilation_w;
  int64_t parallel_imgs_ = (int64_t)parallel_imgs;
  int64_t n_offset_grps_ = (int64_t)n_offset_grps;
  int64_t out_h_ = (int64_t)out_h;
  int64_t out_w_ = (int64_t)out_w;

  id<MTLBuffer> columnBuffer = getMTLBufferStorage(columns_);
  id<MTLBuffer> offsetBuffer = getMTLBufferStorage(offset_);
  id<MTLBuffer> maskBuffer = getMTLBufferStorage(mask_);
  id<MTLBuffer> gradImBuffer = getMTLBufferStorage(grad_im);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool{
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "deformable_col2im_" + scalarToMetalTypeString(columns.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, 
          /*Kernel Name*/ kernel, 
          /*Tensorlist*/{columns_, offset_, mask_});

      [computeEncoder setComputePipelineState:visionPSO];

      [computeEncoder setBytes:&num_kernels length:sizeof(int64_t) atIndex:0];
      [computeEncoder setBuffer:columnBuffer offset:columns_.storage_offset() * columns_.element_size() atIndex:1];
      [computeEncoder setBuffer:offsetBuffer offset:offset_.storage_offset() * offset_.element_size() atIndex:2];
      [computeEncoder setBuffer:maskBuffer offset:mask_.storage_offset() * mask_.element_size() atIndex:3];
      [computeEncoder setBytes:&channels_ length:sizeof(int64_t) atIndex:4];
      [computeEncoder setBytes:&height_ length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&width_ length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&weight_h_ length:sizeof(int64_t) atIndex:7];
      [computeEncoder setBytes:&weight_w_ length:sizeof(int64_t) atIndex:8];
      [computeEncoder setBytes:&pad_h_ length:sizeof(int64_t) atIndex:9];
      [computeEncoder setBytes:&pad_w_ length:sizeof(int64_t) atIndex:10];
      [computeEncoder setBytes:&stride_h_ length:sizeof(int64_t) atIndex:11];
      [computeEncoder setBytes:&stride_w_ length:sizeof(int64_t) atIndex:12];
      [computeEncoder setBytes:&dilation_h_ length:sizeof(int64_t) atIndex:13];
      [computeEncoder setBytes:&dilation_w_ length:sizeof(int64_t) atIndex:14];
      [computeEncoder setBytes:&parallel_imgs_ length:sizeof(int64_t) atIndex:15];
      [computeEncoder setBytes:&n_offset_grps_ length:sizeof(int64_t) atIndex:16];
      [computeEncoder setBytes:&out_h_ length:sizeof(int64_t) atIndex:17];
      [computeEncoder setBytes:&out_w_ length:sizeof(int64_t) atIndex:18];
      [computeEncoder setBytes:&use_mask length:sizeof(bool) atIndex:19];
      [computeEncoder setBuffer:gradImBuffer offset:grad_im.storage_offset() * grad_im.element_size() atIndex:20];

      NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > threadsPerBlock) {
        tgSize = threadsPerBlock;
      }
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);

      NSUInteger gridSize = static_cast<NSUInteger>((num_kernels + tgSize - 1) / tgSize);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSize, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });
}

void compute_grad_offset_and_mask(
    const at::Tensor& columns,
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    int channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int parallel_imgs,
    int n_offset_grps,
    bool use_mask,
    at::Tensor grad_offset,
    at::Tensor grad_mask) {
  using namespace at::native::mps;

  const int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  const int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  const int64_t num_kernels = (int64_t)out_h * out_w * 2 * weight_h * weight_w *
      n_offset_grps * parallel_imgs;

  auto columns_ = columns.contiguous();
  auto input_ = input.contiguous();
  auto offset_ = offset.contiguous();
  auto mask_ = mask.contiguous();

  int64_t channels_ = (int64_t)channels;
  int64_t height_ = (int64_t)height;
  int64_t width_ = (int64_t)width;
  int64_t weight_h_ = (int64_t)weight_h;
  int64_t weight_w_ = (int64_t)weight_w;
  int64_t pad_h_ = (int64_t)pad_h;
  int64_t pad_w_ = (int64_t)pad_w;
  int64_t stride_h_ = (int64_t)stride_h;
  int64_t stride_w_ = (int64_t)stride_w;
  int64_t dilation_h_ = (int64_t)dilation_h;
  int64_t dilation_w_ = (int64_t)dilation_w;
  int64_t parallel_imgs_ = (int64_t)parallel_imgs;
  int64_t n_offset_grps_ = (int64_t)n_offset_grps;
  int64_t out_h_ = (int64_t)out_h;
  int64_t out_w_ = (int64_t)out_w;
  const int64_t offset_channles = (int64_t)2 * weight_h * weight_w * n_offset_grps;
  
  id<MTLBuffer> columnBuffer = getMTLBufferStorage(columns_);
  id<MTLBuffer> inputBuffer = getMTLBufferStorage(input_);
  id<MTLBuffer> offsetBuffer = getMTLBufferStorage(offset_);
  id<MTLBuffer> maskBuffer = getMTLBufferStorage(mask_);
  id<MTLBuffer> gradOffestBuffer = getMTLBufferStorage(grad_offset);
  id<MTLBuffer> gradMaskBuffer = getMTLBufferStorage(grad_mask);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool{
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      const std::string kernel = "deformable_col2im_coord_" + scalarToMetalTypeString(columns.scalar_type());
      id<MTLComputePipelineState> visionPSO = mps::visionPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(visionPSO, 
          /*Kernel Name*/ kernel, 
          /*Tensorlist*/{columns_, input_, offset_, mask_});

      [computeEncoder setComputePipelineState:visionPSO];

      [computeEncoder setBytes:&num_kernels length:sizeof(int64_t) atIndex:0];
      [computeEncoder setBuffer:columnBuffer offset:columns_.storage_offset() * columns_.element_size() atIndex:1];
      [computeEncoder setBuffer:inputBuffer offset:input_.storage_offset() * input_.element_size() atIndex:2];
      [computeEncoder setBuffer:offsetBuffer offset:offset_.storage_offset() * offset_.element_size() atIndex:3];
      [computeEncoder setBuffer:maskBuffer offset:mask_.storage_offset() * mask_.element_size() atIndex:4];
      [computeEncoder setBytes:&channels_ length:sizeof(int64_t) atIndex:5];
      [computeEncoder setBytes:&height_ length:sizeof(int64_t) atIndex:6];
      [computeEncoder setBytes:&width_ length:sizeof(int64_t) atIndex:7];
      [computeEncoder setBytes:&weight_h_ length:sizeof(int64_t) atIndex:8];
      [computeEncoder setBytes:&weight_w_ length:sizeof(int64_t) atIndex:9];
      [computeEncoder setBytes:&pad_h_ length:sizeof(int64_t) atIndex:10];
      [computeEncoder setBytes:&pad_w_ length:sizeof(int64_t) atIndex:11];
      [computeEncoder setBytes:&stride_h_ length:sizeof(int64_t) atIndex:12];
      [computeEncoder setBytes:&stride_w_ length:sizeof(int64_t) atIndex:13];
      [computeEncoder setBytes:&dilation_h_ length:sizeof(int64_t) atIndex:14];
      [computeEncoder setBytes:&dilation_w_ length:sizeof(int64_t) atIndex:15];
      [computeEncoder setBytes:&parallel_imgs_ length:sizeof(int64_t) atIndex:16];
      [computeEncoder setBytes:&offset_channles length:sizeof(int64_t) atIndex:17];
      [computeEncoder setBytes:&n_offset_grps_ length:sizeof(int64_t) atIndex:18];
      [computeEncoder setBytes:&out_h_ length:sizeof(int64_t) atIndex:19];
      [computeEncoder setBytes:&out_w_ length:sizeof(int64_t) atIndex:20];
      [computeEncoder setBytes:&use_mask length:sizeof(bool) atIndex:21];
      [computeEncoder setBuffer:gradOffestBuffer offset:grad_offset.storage_offset() * grad_offset.element_size() atIndex:22];
      [computeEncoder setBuffer:gradMaskBuffer offset:grad_mask.storage_offset() * grad_mask.element_size() atIndex:23];

      NSUInteger tgSize = visionPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > threadsPerBlock) {
        tgSize = threadsPerBlock;
      }
      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);

      NSUInteger gridSize = static_cast<NSUInteger>((num_kernels + tgSize - 1) / tgSize);
      MTLSize threadgroupsPerGrid = MTLSizeMake(gridSize, 1, 1);

      [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadGroupSize];

      getMPSProfiler().endProfileKernel(visionPSO);
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_gradient_inputs(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor grad_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int n_weight_grps,
    int n_offset_grps,
    int n_parallel_imgs,
    bool use_mask) {
  at::DeviceGuard guard(input.device());

  int batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  long out_w =
      (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  long out_h =
      (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;

  auto grad_input = at::zeros_like(input);
  auto grad_offset = at::zeros_like(offset);
  auto grad_mask = at::zeros_like(mask);

  if (batch_sz == 0) {
    return std::make_tuple(grad_input, grad_offset, grad_mask);
  }

  auto columns = at::empty(
      {n_in_channels * weight_w * weight_h, n_parallel_imgs * out_h * out_w},
      input.options());

  // Separate into blocks
  grad_input = grad_input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
  input = input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  grad_offset = grad_offset.reshape(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});
  offset = offset.reshape(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    grad_mask = grad_mask.reshape(
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
    mask = mask.reshape(
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  grad_out = grad_out
                 .reshape(
                     {batch_sz / n_parallel_imgs,
                      n_parallel_imgs,
                      n_weight_grps,
                      n_out_channels / n_weight_grps,
                      out_h,
                      out_w})
                 .permute({0, 2, 3, 1, 4, 5});

  weight = weight.reshape(
      {n_weight_grps,
       weight.size(0) / n_weight_grps,
       weight.size(1),
       weight.size(2),
       weight.size(3)});

  columns = columns.view(
      {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    columns.zero_();
    // Separate into weight groups
    for (int g = 0; g < n_weight_grps; g++) {
      columns[g] = columns[g].addmm_(
          weight[g].flatten(1).transpose(0, 1), grad_out[elt][g].flatten(1));
    }

    compute_grad_offset_and_mask(
        columns,
        input[elt],
        offset[elt],
        mask[elt],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        grad_offset[elt],
        grad_mask[elt]);

    compute_grad_input(
        columns,
        offset[elt],
        mask[elt],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        grad_input[elt]);
  }

  grad_input = grad_input.view({batch_sz, n_in_channels, in_h, in_w});
  grad_offset = grad_offset.view(
      {batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    grad_mask = grad_mask.view(
        {batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  return std::make_tuple(grad_input, grad_offset, grad_mask);
}

at::Tensor backward_gradient_parameters(
    at::Tensor input,
    const at::Tensor& weight,
    at::Tensor offset,
    at::Tensor mask,
    const at::Tensor& grad_out,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int n_weight_grps,
    int n_offset_grps,
    int n_parallel_imgs,
    bool use_mask) {
  at::DeviceGuard guard(input.device());

  int batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  long out_h = grad_out.size(2);
  long out_w = grad_out.size(3);

  auto grad_weight = at::zeros_like(weight);
  if (batch_sz == 0) {
    return grad_weight;
  }

  at::Tensor grad_out_buf = grad_out
                                .reshape(
                                    {batch_sz / n_parallel_imgs,
                                     n_parallel_imgs,
                                     n_weight_grps,
                                     n_out_channels / n_weight_grps,
                                     out_h,
                                     out_w})
                                .permute({0, 2, 3, 1, 4, 5})
                                .contiguous();

  input = input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  offset = offset.reshape(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    mask = mask.reshape(
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  grad_weight = grad_weight.reshape(
      {n_weight_grps,
       grad_weight.size(0) / n_weight_grps,
       grad_weight.size(1),
       grad_weight.size(2),
       grad_weight.size(3)});

  auto columns = at::empty(
      {n_weight_grps,
       n_in_channels * weight_w * weight_h / n_weight_grps,
       n_parallel_imgs * out_h * out_w},
      input.options());

  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    deformable_im2col(
        input[elt],
        offset[elt],
        mask[elt],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        columns);

    for (int g = 0; g < n_weight_grps; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(
                  grad_out_buf[elt][g].flatten(1), columns[g].transpose(1, 0))
              .view_as(grad_weight[g]);
    }
  }

  grad_weight = grad_weight.view(
      {grad_weight.size(0) * grad_weight.size(1),
       grad_weight.size(2),
       grad_weight.size(3),
       grad_weight.size(4)});
  return grad_weight;
}

at::Tensor deform_conv2d_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  at::Tensor input_c = input.contiguous();
  at::Tensor offset_c = offset.contiguous();
  at::Tensor weight_c = weight.contiguous();
  at::Tensor mask_c = mask.contiguous();
  at::Tensor bias_c = bias.contiguous();

  TORCH_CHECK(input_c.ndimension() == 4);
  TORCH_CHECK(offset_c.ndimension() == 4);
  TORCH_CHECK(!use_mask || mask_c.ndimension() == 4);
  TORCH_CHECK(weight_c.ndimension() == 4);
  TORCH_CHECK(input_c.is_mps(), "input must be a MPS tensor");

  at::DeviceGuard guard(input_c.device());

  int batch_sz = input_c.size(0);
  int in_channels = input_c.size(1);
  int in_h = input_c.size(2);
  int in_w = input_c.size(3);

  int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  int out_channels = weight_c.size(0);
  int weight_h = weight_c.size(2);
  int weight_w = weight_c.size(3);

  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(
      weight_h > 0 && weight_w > 0,
      "weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);
  TORCH_CHECK(
      stride_h > 0 && stride_w > 0,
      "stride_h: ",
      stride_h,
      " stride_w: ",
      stride_w);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  TORCH_CHECK(
      dilation_h > 0 && dilation_w > 0,
      "dilation_h: ",
      dilation_h,
      " dilation_w: ",
      dilation_w);

  TORCH_CHECK(weight_c.size(1) * n_weight_grps == input_c.size(1));
  TORCH_CHECK(weight_c.size(0) % n_weight_grps == 0);
  TORCH_CHECK(
      (offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset_c.size(1),
      " expected: ",
      n_offset_grps * 2 * weight_h * weight_w);
  TORCH_CHECK(
      (!use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask_c.size(1),
      " expected: ",
      n_offset_grps * weight_h * weight_w);
  TORCH_CHECK(input_c.size(1) % n_offset_grps == 0);

  TORCH_CHECK(
      (offset_c.size(0) == input_c.size(0)), "invalid batch size of offset");
  TORCH_CHECK(
      (offset_c.size(2) == out_h && offset_c.size(3) == out_w),
      "offset output dims: (",
      offset_c.size(2),
      ", ",
      offset_c.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      (mask_c.size(0) == input_c.size(0)), "invalid batch size of mask");
  TORCH_CHECK(
      (!use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w)),
      "mask output dims: (",
      mask_c.size(2),
      ", ",
      mask_c.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);

  auto out =
      at::zeros({batch_sz, out_channels, out_h, out_w}, input_c.options());
  if (batch_sz == 0) {
    return out;
  }

  // Separate batches into blocks
  out = out.view(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       out_channels,
       out_h,
       out_w});
  input_c = input_c.view(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_h, in_w});

  offset_c = offset_c.view(
      {batch_sz / n_parallel_imgs,
       n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w,
       out_h,
       out_w});

  if (use_mask) {
    mask_c = mask_c.view(
        {batch_sz / n_parallel_imgs,
         n_parallel_imgs,
         n_offset_grps * weight_h * weight_w,
         out_h,
         out_w});
  }

  at::Tensor out_buf = at::zeros(
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs * out_h,
       out_w},
      out.options());
  
  // Separate channels into convolution groups
  out_buf = out_buf.view(
      {out_buf.size(0),
       n_weight_grps,
       out_buf.size(1) / n_weight_grps,
       out_buf.size(2),
       out_buf.size(3)});
  weight_c = weight_c.view(
      {n_weight_grps,
       weight_c.size(0) / n_weight_grps,
       weight_c.size(1),
       weight_c.size(2),
       weight_c.size(3)});

  // Sample points and perform convolution
  auto columns = at::zeros(
      {in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w},
      input_c.options());
  for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
    deformable_im2col(
        input_c[b],
        offset_c[b],
        mask_c[b],
        in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        use_mask,
        columns);
    columns = columns.view(
        {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      out_buf[b][g] = out_buf[b][g]
                          .flatten(1)
                          .addmm_(weight_c[g].flatten(1), columns[g])
                          .view_as(out_buf[b][g]);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  out_buf = out_buf.view(
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs,
       out_h,
       out_w});
  out_buf.transpose_(1, 2);
  out.copy_(out_buf);
  out = out.view({batch_sz, out_channels, out_h, out_w});

  return out + bias_c.view({1, out_channels, 1, 1});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
deform_conv2d_backward_kernel(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  at::Tensor grad_out_c = grad_out.contiguous();
  at::Tensor input_c = input.contiguous();
  at::Tensor weight_c = weight.contiguous();
  at::Tensor offset_c = offset.contiguous();
  at::Tensor mask_c = mask.contiguous();
  at::Tensor bias_c = bias.contiguous();

  const int batch_sz = input_c.size(0);
  const int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  auto grad_input_and_offset_and_mask = backward_gradient_inputs(
      input_c,
      weight_c,
      offset_c,
      mask_c,
      grad_out_c,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      n_weight_grps,
      n_offset_grps,
      n_parallel_imgs,
      use_mask);

  auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
  auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
  auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);

  auto grad_weight = backward_gradient_parameters(
      input_c,
      weight_c,
      offset_c,
      mask_c,
      grad_out_c,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      n_weight_grps,
      n_offset_grps,
      n_parallel_imgs,
      use_mask);

  auto value = grad_out_c.sum({0, 2, 3});
  auto grad_bias = at::ones_like(bias_c) * value;

  return std::make_tuple(
      grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, MPS, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN(deform_conv2d_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_deform_conv2d_backward"),
      TORCH_FN(deform_conv2d_backward_kernel));
}

} // namespace ops
} // namespace vision
