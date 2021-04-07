import sys
sys.path.insert(0,'..')

from functools import reduce
import numpy as np
import tensorrt
import pycuda.driver as cuda
import pycuda.autoinit
from trt_lite import TrtLite
import mmcv
import ctypes
import torch
import time

torch.cuda.init()
ctypes.cdll.LoadLibrary('./DeformConvPlugin.so')

def generate_frame_indices(crt_idx,
                           max_frame_num,
                           num_frames,
                           padding='reflection'):
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle',
                       'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices
 
def run_engine_dynamic(input_data):
#    trt = TrtLite(build_engine_dynamic)
#    trt.print_info()
#    trt.save_to_file("edvr.trt")

    trt = TrtLite(engine_file_path="edvr.trt")
    trt.print_info()

    io_info = trt.get_io_info({})
    if io_info is None:
        return
    print(io_info)
    h_buffers = trt.allocate_io_buffers({}, False)
    d_buffers = trt.allocate_io_buffers({}, True)
    
    h_buffers[0][:] = input_data

    for i, info in enumerate(io_info):
        if info[1]:
            cuda.memcpy_htod(d_buffers[i], h_buffers[i])
    trt.execute(d_buffers, {})

    nRound = 10
    cuda.Context.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute(d_buffers, {})
    cuda.Context.synchronize()
    print('Prediction time: ', (time.time() - t0) / nRound)

    for i, info in enumerate(io_info):
        if not info[1]:
            cuda.memcpy_dtoh(h_buffers[i], d_buffers[i])

    name2tensor = {info[0]:h_buffers[i] for i,info in enumerate(io_info) if not info[1]}
    np.savez('out.npz', **name2tensor)

if __name__ == '__main__':
    dir_list = ['/home/gji/gji/DLmodels/EDVR/datasets/Vid4/BIx4/calendar/frame_00{:02d}.png'.format(i) for i in range(3, 11)]
    img_paths = [dir_list[i] for i in generate_frame_indices(0, 8, 7, 'reflection_circle')]
    input_data = np.asarray([mmcv.bgr2rgb(mmcv.imread(v)).astype(np.float32).transpose((2, 0, 1)) / 255.0 for v in img_paths])[np.newaxis, ...]
    print('input_data: ', input_data.shape)
    run_engine_dynamic(input_data)
