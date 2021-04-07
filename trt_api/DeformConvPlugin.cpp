// g++ -fPIC DeformConvPlugin.cpp -o DeformConvPlugin.so -g -shared -isystem /usr/local/lib/python3.6/dist-packages/torch/include  -isystem /usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include  -isystem /usr/local/tensorrt7/include -L/usr/local/tensorrt7/lib -lnvinfer

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

int DeformConvPlugin::enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    at::Tensor 
        input = torch::from_blob(const_cast<void *>(inputs[0]), {1, 128, m.inputDim.d[2], m.inputDim.d[3]}, options),
        weight = torch::from_blob(const_cast<void *>(inputs[1]), {128, 128, 3, 3}, options),
        bias = torch::from_blob(const_cast<void *>(inputs[2]), {128}, options),
        ones = at::Tensor(),
        offset = torch::from_blob(const_cast<void *>(inputs[3]), {1, 144, m.inputDim.d[2], m.inputDim.d[3]}, options),
        mask = torch::from_blob(const_cast<void *>(inputs[4]), {1, 72, m.inputDim.d[2], m.inputDim.d[3]}, options),
        output = torch::from_blob(outputs[0], {1, 128, m.inputDim.d[2], m.inputDim.d[3]}, options),
        columns = at::Tensor();

    modulated_deform_conv_forward(input, weight, bias, ones, offset, mask, output, columns, 3, 3, 1, 1, 1, 1, 1, 1, 1, 8, true);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(DeformConvPluginCreator);

