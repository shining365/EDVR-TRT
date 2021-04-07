/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeformConvPlugin.h"
#include <chrono>
#include <thread>
#include <torch/torch.h>
#include <c10/core/Device.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

void modulated_deform_conv_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias);

int DeformConvPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
    nvinfer1::Dims inputDim = inputDesc[0].dims;
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    at::Tensor 
        input = torch::from_blob(const_cast<void *>(inputs[0]), {1, 128, inputDim.d[2], inputDim.d[3]}, options),
        weight = torch::from_blob(const_cast<void *>(inputs[1]), {128, 128, 3, 3}, options),
        bias = torch::from_blob(const_cast<void *>(inputs[2]), {128}, options),
        ones = at::Tensor(),
        offset = torch::from_blob(const_cast<void *>(inputs[3]), {1, 144, inputDim.d[2], inputDim.d[3]}, options),
        mask = torch::from_blob(const_cast<void *>(inputs[4]), {1, 72, inputDim.d[2], inputDim.d[3]}, options),
        output = torch::from_blob(outputs[0], {1, 128, inputDim.d[2], inputDim.d[3]}, options),
        columns = at::Tensor();

    modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output, columns, 3, 3, 1, 1, 1, 1, 1, 1, 1, 8, true);
    return 0;
}

nvinfer1::PluginFieldCollection DeformConvPluginCreator::fc;
REGISTER_TENSORRT_PLUGIN(DeformConvPluginCreator);
