#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/DeviceGuard.h>
#include <THC/THCAtomics.cuh>

#include <cmath>
#include <vector>

#define WITH_CUDA

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}


template <typename scalar_t>
__device__ scalar_t dmcn_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
                                         const int height, const int width, scalar_t h, scalar_t w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
						       const scalar_t *data_im, const scalar_t *data_offset, const scalar_t *data_mask,
						       const int height, const int width, const int kernel_h, const int kernel_w,
						       const int pad_h, const int pad_w,
						       const int stride_h, const int stride_w,
						       const int dilation_h, const int dilation_w,
						       const int channel_per_deformable_group,
						       const int batch_size, const int num_channels, const int deformable_group,
						       const int height_col, const int width_col,
						       scalar_t *data_col)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		// index index of output matrix
		const int w_col = index % width_col;
		const int h_col = (index / width_col) % height_col;
		const int b_col = (index / width_col / height_col) % batch_size;
		const int c_im = (index / width_col / height_col) / batch_size;
		const int c_col = c_im * kernel_h * kernel_w;

		// compute deformable group index
		const int deformable_group_index = c_im / channel_per_deformable_group;

		const int h_in = h_col * stride_h - pad_h;
		const int w_in = w_col * stride_w - pad_w;

		scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
		//const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
		const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
		const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

		const scalar_t *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

		for (int i = 0; i < kernel_h; ++i)
		{
			for (int j = 0; j < kernel_w; ++j)
			{
				const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
				const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
				const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
				const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
				const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
				const scalar_t mask = data_mask_ptr[data_mask_hw_ptr];
				scalar_t val = static_cast<scalar_t>(0);
				const scalar_t h_im = h_in + i * dilation_h + offset_h;
				const scalar_t w_im = w_in + j * dilation_w + offset_w;
				//if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
				if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
				{
					//const float map_h = i * dilation_h + offset_h;
					//const float map_w = j * dilation_w + offset_w;
					//const int cur_height = height - h_in;
					//const int cur_width = width - w_in;
					//val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
					val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
				}
				*data_col_ptr = val * mask;
				data_col_ptr += batch_size * height_col * width_col;
				//data_col_ptr += height_col * width_col;
			}
		}
	}
}

void modulated_deformable_im2col_cuda(
    const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im, const int width_im,
    const int height_col, const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int deformable_group, at::Tensor data_col)
{
	// num_axes should be smaller than block size
	const int channel_per_deformable_group = channels / deformable_group;
	const int num_kernels = channels * batch_size * height_col * width_col;

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(
	    data_im.scalar_type(), "modulated_deformable_im2col_gpu", ([&] {
		    const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
		    const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
		    const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
		    scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

		    modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
			num_kernels, data_im_, data_offset_, data_mask_, height_im, width_im, kernel_h, kenerl_w,
			pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
			batch_size, channels, deformable_group, height_col, width_col, data_col_);
	    }));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
	}
}

void modulated_deform_conv_cuda_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias)
{
	TORCH_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
	TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
	at::DeviceGuard guard(input.device());

	const int batch = input.size(0);
	const int channels = input.size(1);
	const int height = input.size(2);
	const int width = input.size(3);

	const int channels_out = weight.size(0);
	const int channels_kernel = weight.size(1);
	const int kernel_h_ = weight.size(2);
	const int kernel_w_ = weight.size(3);

	if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
		AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
			 kernel_h_, kernel_w, kernel_h_, kernel_w_);
	if (channels != channels_kernel * group)
		AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
			 channels, channels_kernel * group);

	const int height_out =
	    (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int width_out =
	    (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	// if (ones.ndimension() != 2 ||
	//     ones.size(0) * ones.size(1) < height_out * width_out)
	// {
		// Resize plane and fill with ones...
		ones = at::ones({height_out, width_out}, input.options());
	// }

	// resize output
	output = output.view({batch, channels_out, height_out, width_out}).zero_();
	// resize temporary columns
	columns =
	    at::zeros({channels * kernel_h * kernel_w, 1 * height_out * width_out},
		      input.options());

	output = output.view({output.size(0), group, output.size(1) / group,
			      output.size(2), output.size(3)});

	for (int b = 0; b < batch; b++)
	{
		modulated_deformable_im2col_cuda(
		    input[b], offset[b], mask[b], 1, channels, height, width, height_out,
		    width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
		    dilation_h, dilation_w, deformable_group, columns);

		// divide into group
		weight = weight.view({group, weight.size(0) / group, weight.size(1),
				      weight.size(2), weight.size(3)});
		columns = columns.view({group, columns.size(0) / group, columns.size(1)});

		for (int g = 0; g < group; g++)
		{
			output[b][g] = output[b][g]
					   .flatten(1)
					   .addmm_(weight[g].flatten(1), columns[g])
					   .view_as(output[b][g]);
		}

		weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
				      weight.size(3), weight.size(4)});
		columns =
		    columns.view({columns.size(0) * columns.size(1), columns.size(2)});
	}

	output = output.view({output.size(0), output.size(1) * output.size(2),
			      output.size(3), output.size(4)});

	if (with_bias)
	{
		output += bias.view({1, bias.size(0), 1, 1});
	}
}

void modulated_deform_conv_forward(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor ones,
    at::Tensor offset, at::Tensor mask, at::Tensor output, at::Tensor columns,
    int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int deformable_group,
    const bool with_bias)
{
	if (input.device().is_cuda())
	{
#ifdef WITH_CUDA
		return modulated_deform_conv_cuda_forward(input, weight, bias, ones,
							  offset, mask, output, columns, kernel_h, kernel_w, stride_h,
							  stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
							  deformable_group, with_bias);
#else
		AT_ERROR("modulated deform conv is not compiled with GPU support");
#endif
	}
	AT_ERROR("modulated deform conv is not implemented on CPU");
}

//nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -isystem /usr/include/python3.6m/ -isystem /usr/local/lib/python3.6/dist-packages/torch/include -isystem /usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include -L /usr/local/lib/python3.6/dist-packages/torch/lib/ -lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda deform_conv.cu
