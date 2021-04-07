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

def get_plugin_creator(plugin_name):
    plugin_creator_list = tensorrt.get_plugin_registry().plugin_creator_list
    plugin_creator = None
    for c in plugin_creator_list:
        if c.name == plugin_name:
            plugin_creator = c
    return plugin_creator

#np.set_printoptions(threshold=np.inf)

def make_layer(network, weights, last_layer, num_feat, n_block, prefix):
    for i in range(n_block):
        conv = network.add_convolution(last_layer.get_output(0), num_feat, (3, 3), 
            weights[prefix + f'.{i}.conv1.weight'], 
            weights[prefix + f'.{i}.conv1.bias'])
        conv.stride = (1, 1)
        conv.padding = (1, 1)
        
        act = network.add_activation(conv.get_output(0), tensorrt.ActivationType.RELU)
        
        conv = network.add_convolution(act.get_output(0), num_feat, (3, 3), 
            weights[prefix + f'.{i}.conv2.weight'], 
            weights[prefix + f'.{i}.conv2.bias'])
        conv.stride = (1, 1)
        conv.padding = (1, 1)

        last_layer = network.add_elementwise(last_layer.get_output(0), conv.get_output(0), tensorrt.ElementWiseOperation.SUM)
    
    return last_layer

def dcn_pack(network, weights, feat, offset, name_prefix, dim2):
    out = network.add_convolution(offset.get_output(0), 72 * 3, (3, 3), weights[name_prefix + '.conv_offset.weight'], weights[name_prefix + '.conv_offset.bias'])
    out.stride = (1, 1)
    out.padding = (1, 1)
    offset = network.add_slice(out.get_output(0), (0, 0, 0, 0, 0), (1, 1, 72 * 2) + dim2, (1, 1, 1, 1, 1))
    mask = network.add_slice(out.get_output(0), (0, 0, 72 * 2, 0, 0), (1, 1, 72) + dim2, (1, 1, 1, 1, 1))

    weight = network.add_constant(weights[name_prefix + '.weight'].shape, weights[name_prefix + '.weight'])
    bias = network.add_constant(weights[name_prefix + '.bias'].shape, weights[name_prefix + '.bias'])
    plugin_creator = get_plugin_creator('DeformConvPlugin')
    if plugin_creator == None:
        print('Plugin DeformConvPlugin not found. Exiting')
        exit()
    print('plugin input', feat.get_output(0).shape)
    print('plugin weight', weight.get_output(0).shape)
    print('plugin bias', bias.get_output(0).shape)
    print('plugin offset', offset.get_output(0).shape)
    print('plugin mask', mask.get_output(0).shape)
    return network.add_plugin_v2([feat.get_output(0), weight.get_output(0), bias.get_output(0), offset.get_output(0), mask.get_output(0)], 
        plugin_creator.create_plugin('DeformConvPlugin', tensorrt.PluginFieldCollection()))

def pcd_alignment(network, weights, num_feat, nbr_feat_l, ref_feat_l):
    upsampled_offset, upsampled_feat = None, None
    for l in range(3, 0, -1):
        offset = network.add_concatenation([nbr_feat_l[l - 1].get_output(0), ref_feat_l[l - 1].get_output(0)])
        offset.axis = 2

        offset = network.add_convolution(offset.get_output(0), num_feat, (3, 3), 
            weights[f'pcd_align.offset_conv1.l{l}.weight'], 
            weights[f'pcd_align.offset_conv1.l{l}.bias'])
        offset.stride = (1, 1)
        offset.padding = (1, 1)
        offset = network.add_activation(offset.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
        offset.alpha = 0.1
        print(f'l{l}_' + 'conv1', offset.get_output(0).shape)

        if l < 3:
            offset = network.add_concatenation([offset.get_output(0), upsampled_offset.get_output(0)])
            offset.axis = 2

        offset = network.add_convolution(offset.get_output(0), num_feat, (3, 3), 
            weights[f'pcd_align.offset_conv2.l{l}.weight'], 
            weights[f'pcd_align.offset_conv2.l{l}.bias'])
        offset.stride = (1, 1)
        offset.padding = (1, 1)
        offset = network.add_activation(offset.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
        offset.alpha = 0.1
        print(f'l{l}_' + 'conv2', offset.get_output(0).shape)

        if l < 3:
            offset = network.add_convolution(offset.get_output(0), num_feat, (3, 3), 
                weights[f'pcd_align.offset_conv3.l{l}.weight'], 
                weights[f'pcd_align.offset_conv3.l{l}.bias'])
            offset.stride = (1, 1)
            offset.padding = (1, 1)
            offset = network.add_activation(offset.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
            offset.alpha = 0.1
            print(f'l{l}_' + 'conv3', offset.get_output(0).shape)

        feat = dcn_pack(network, weights, nbr_feat_l[l - 1], offset, f'pcd_align.dcn_pack.l{l}', (36, 44))

        if l < 3:
            concat = network.add_concatenation([feat.get_output(0), upsampled_feat.get_output(0)])
            concat.axis = 2

            feat = network.add_convolution(concat.get_output(0), num_feat, (3, 3), 
                weights[f'pcd_align.feat_conv.l{l}.weight'], 
                weights[f'pcd_align.feat_conv.l{l}.bias'])
            feat.stride = (1, 1)
            feat.padding = (1, 1)
            print(f'l{l}_' + 'feat_conv', feat.get_output(0).shape)

        if l > 1:
            upsampled_offset = network.add_resize(offset.get_output(0))
            upsampled_offset.resize_mode = tensorrt.ResizeMode.LINEAR
            upsampled_offset.scales = [1.0, 1.0, 1.0, 2.0, 2.0]

            two = network.add_constant((1, 1, 1, 1, 1), np.asarray([2.0], dtype=np.float32))
            upsampled_offset = network.add_elementwise(upsampled_offset.get_output(0), two.get_output(0), tensorrt.ElementWiseOperation.PROD)

            feat = network.add_activation(feat.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
            feat.alpha = 0.1

            upsampled_feat = network.add_resize(feat.get_output(0))
            upsampled_feat.resize_mode = tensorrt.ResizeMode.LINEAR
            upsampled_feat.scales = [1.0, 1.0, 1.0, 2.0, 2.0]

    offset = network.add_concatenation([feat.get_output(0), ref_feat_l[0].get_output(0)])
    offset.axis = 2

    offset = network.add_convolution(offset.get_output(0), num_feat, (3, 3), 
        weights['pcd_align.cas_offset_conv1.weight'], 
        weights['pcd_align.cas_offset_conv1.bias'])
    offset.stride = (1, 1)
    offset.padding = (1, 1)
    offset = network.add_activation(offset.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    offset.alpha = 0.1
    print('cas_offset_conv1', offset.get_output(0).shape)

    offset = network.add_convolution(offset.get_output(0), num_feat, (3, 3), 
        weights['pcd_align.cas_offset_conv2.weight'], 
        weights['pcd_align.cas_offset_conv2.bias'])
    offset.stride = (1, 1)
    offset.padding = (1, 1)
    offset = network.add_activation(offset.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    offset.alpha = 0.1
    print('cas_offset_conv2', offset.get_output(0).shape)
    print('feat', feat.get_output(0).shape)

    feat = dcn_pack(network, weights, feat, offset, 'pcd_align.cas_dcnpack', (144, 176))

    feat = network.add_activation(feat.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat.alpha = 0.1
    print('cas_dcnpack', feat.get_output(0).shape)
    return feat

def tsa(network, weights, num_feat, aligned_feat, n_frame):
    i_center = n_frame // 2

    embedding_ref = network.add_slice(aligned_feat.get_output(0), (0, i_center, 0, 0, 0), (1, 1, 128, 144, 176), (1, 1, 1, 1, 1))
    embedding_ref = network.add_convolution(embedding_ref.get_output(0), num_feat, (3, 3), weights['fusion.temporal_attn1.weight'], weights['fusion.temporal_attn1.bias'])
    embedding_ref.stride = (1, 1)
    embedding_ref.padding = (1, 1)
    print('embedding_ref', embedding_ref.get_output(0).shape)

    embedding = network.add_convolution(aligned_feat.get_output(0), num_feat, (3, 3), weights['fusion.temporal_attn2.weight'], weights['fusion.temporal_attn2.bias'])
    embedding.stride = (1, 1)
    embedding.padding = (1, 1)
    print('embedding', embedding.get_output(0).shape)

    corr_prob = network.add_elementwise(embedding_ref.get_output(0), embedding.get_output(0), tensorrt.ElementWiseOperation.PROD)
    corr_prob = network.add_reduce(corr_prob.get_output(0), tensorrt.ReduceOperation.SUM, 0b100, True)
    corr_prob = network.add_activation(corr_prob.get_output(0), tensorrt.ActivationType.SIGMOID)
    zeros = network.add_constant((1, 1, 128, 1, 1), np.zeros(128, dtype=np.float32))
    corr_prob = network.add_elementwise(corr_prob.get_output(0), zeros.get_output(0), tensorrt.ElementWiseOperation.SUM)
    print('corr_prob', corr_prob.get_output(0).shape)

    aligned_feat = network.add_elementwise(aligned_feat.get_output(0), corr_prob.get_output(0), tensorrt.ElementWiseOperation.PROD)
    aligned_feat = network.add_shuffle(aligned_feat.get_output(0))
    aligned_feat.reshape_dims = (1, n_frame * 128, 144, 176)
    print('aligned_feat', aligned_feat.get_output(0).shape)

    feat = network.add_convolution(aligned_feat.get_output(0), num_feat, (1, 1), weights['fusion.feat_fusion.weight'], weights['fusion.feat_fusion.bias'])
    feat = network.add_activation(feat.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat.alpha = 0.1
    print('feat', feat.get_output(0).shape)

    attn = network.add_convolution(aligned_feat.get_output(0), num_feat, (1, 1), weights['fusion.spatial_attn1.weight'], weights['fusion.spatial_attn1.bias'])
    attn = network.add_activation(attn.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn.alpha = 0.1
    attn_max = network.add_pooling(attn.get_output(0), tensorrt.PoolingType.MAX, (3, 3))
    attn_max.stride = (2, 2)
    attn_max.padding = (1, 1)
    attn_avg = network.add_pooling(attn.get_output(0), tensorrt.PoolingType.AVERAGE, (3, 3))
    attn_avg.stride = (2, 2)
    attn_avg.padding = (1, 1)
    attn_avg.average_count_excludes_padding = False
    attn = network.add_concatenation([attn_max.get_output(0), attn_avg.get_output(0)])
    attn.axis = 1
    attn = network.add_convolution(attn.get_output(0), num_feat, (1, 1), weights['fusion.spatial_attn2.weight'], weights['fusion.spatial_attn2.bias'])
    attn = network.add_activation(attn.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn.alpha = 0.1
    print('attn', attn.get_output(0).shape)

    attn_level = network.add_convolution(attn.get_output(0), num_feat, (1, 1), weights['fusion.spatial_attn_l1.weight'], weights['fusion.spatial_attn_l1.bias'])
    attn_level = network.add_activation(attn_level.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn_level.alpha = 0.1
    attn_level_max = network.add_pooling(attn_level.get_output(0), tensorrt.PoolingType.MAX, (3, 3))
    attn_level_max.stride = (2, 2)
    attn_level_max.padding = (1, 1)
    attn_level_avg = network.add_pooling(attn_level.get_output(0), tensorrt.PoolingType.AVERAGE, (3, 3))
    attn_level_avg.stride = (2, 2)
    attn_level_avg.padding = (1, 1)
    attn_level_avg.average_count_excludes_padding = False
    attn_level = network.add_concatenation([attn_level_max.get_output(0), attn_level_avg.get_output(0)])
    attn_level.axis = 1
    attn_level = network.add_convolution(attn_level.get_output(0), num_feat, (3, 3), weights['fusion.spatial_attn_l2.weight'], weights['fusion.spatial_attn_l2.bias'])
    attn_level.stride = (1, 1)
    attn_level.padding = (1, 1)
    attn_level = network.add_activation(attn_level.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn_level.alpha = 0.1
    print('attn_level', attn_level.get_output(0).shape)

    attn_level = network.add_convolution(attn_level.get_output(0), num_feat, (3, 3), weights['fusion.spatial_attn_l3.weight'], weights['fusion.spatial_attn_l3.bias'])
    attn_level.stride = (1, 1)
    attn_level.padding = (1, 1)
    attn_level = network.add_activation(attn_level.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn_level.alpha = 0.1

    attn_level = network.add_resize(attn_level.get_output(0))
    attn_level.resize_mode = tensorrt.ResizeMode.LINEAR
    attn_level.scales = [1.0, 1.0, 2.0, 2.0]
    print('attn_level', attn_level.get_output(0).shape)

    attn = network.add_convolution(attn.get_output(0), num_feat, (3, 3), weights['fusion.spatial_attn3.weight'], weights['fusion.spatial_attn3.bias'])
    attn.stride = (1, 1)
    attn.padding = (1, 1)
    attn = network.add_activation(attn.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn.alpha = 0.1
    attn = network.add_elementwise(attn.get_output(0), attn_level.get_output(0), tensorrt.ElementWiseOperation.SUM)

    attn = network.add_convolution(attn.get_output(0), num_feat, (1, 1), weights['fusion.spatial_attn4.weight'], weights['fusion.spatial_attn4.bias'])
    attn = network.add_activation(attn.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn.alpha = 0.1
    attn = network.add_resize(attn.get_output(0))
    attn.resize_mode = tensorrt.ResizeMode.LINEAR
    attn.scales = [1.0, 1.0, 2.0, 2.0]
    print('attn', attn.get_output(0).shape)

    attn = network.add_convolution(attn.get_output(0), num_feat, (3, 3), weights['fusion.spatial_attn5.weight'], weights['fusion.spatial_attn5.bias'])
    attn.stride = (1, 1)
    attn.padding = (1, 1)
    print('attn', attn.get_output(0).shape)

    attn_add = network.add_convolution(attn.get_output(0), num_feat, (1, 1), weights['fusion.spatial_attn_add1.weight'], weights['fusion.spatial_attn_add1.bias'])
    attn_add = network.add_activation(attn_add.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    attn_add.alpha = 0.1
    attn_add = network.add_convolution(attn_add.get_output(0), num_feat, (1, 1), weights['fusion.spatial_attn_add2.weight'], weights['fusion.spatial_attn_add2.bias'])
    
    attn = network.add_activation(attn.get_output(0), tensorrt.ActivationType.SIGMOID)

    feat = network.add_elementwise(feat.get_output(0), attn.get_output(0), tensorrt.ElementWiseOperation.PROD)
    two = network.add_constant((1, 1, 1, 1), np.asarray([2.0], dtype=np.float32))
    feat = network.add_elementwise(feat.get_output(0), two.get_output(0), tensorrt.ElementWiseOperation.PROD)
    feat = network.add_elementwise(feat.get_output(0), attn_add.get_output(0), tensorrt.ElementWiseOperation.SUM)
    
    return feat

def build_engine_dynamic(builder):
    network = builder.create_network(1)
    n_frame = 7
    x = network.add_input("x", tensorrt.DataType.FLOAT, (1, n_frame, 3, 144, 176))

    weights = np.load('edvr.npz')

    num_feat = 128
    conv_first = network.add_convolution(x, num_feat, (3, 3), weights['conv_first.weight'], weights['conv_first.bias'])
    conv_first.stride = (1, 1)
    conv_first.padding = (1, 1)
    print('conv_first', conv_first.get_output(0).shape)

    feat_l1 = network.add_activation(conv_first.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat_l1.alpha = 0.1
    
    feat_l1 = make_layer(network, weights, feat_l1, num_feat, 5, 'feature_extraction')
    print('feat_l1', feat_l1.get_output(0).shape)
    
    conv = network.add_convolution(feat_l1.get_output(0), num_feat, (3, 3), weights['conv_l2_1.weight'], weights['conv_l2_1.bias'])
    conv.stride = (2, 2)
    conv.padding = (1, 1)
    feat_l2 = network.add_activation(conv.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat_l2.alpha = 0.1
    
    conv = network.add_convolution(feat_l2.get_output(0), num_feat, (3, 3), weights['conv_l2_2.weight'], weights['conv_l2_2.bias'])
    conv.stride = (1, 1)
    conv.padding = (1, 1)
    feat_l2 = network.add_activation(conv.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat_l2.alpha = 0.1
    print('feat_l2', feat_l2.get_output(0).shape)
    
    conv = network.add_convolution(feat_l2.get_output(0), num_feat, (3, 3), weights['conv_l3_1.weight'], weights['conv_l3_1.bias'])
    conv.stride = (2, 2)
    conv.padding = (1, 1)
    feat_l3 = network.add_activation(conv.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat_l3.alpha = 0.1

    conv = network.add_convolution(feat_l3.get_output(0), num_feat, (3, 3), weights['conv_l3_2.weight'], weights['conv_l3_2.bias'])
    conv.stride = (1, 1)
    conv.padding = (1, 1)
    feat_l3 = network.add_activation(conv.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    feat_l3.alpha = 0.1
    print('feat_l3', feat_l3.get_output(0).shape)

    i_center = n_frame // 2
    ref_feat_l = [
        network.add_slice(feat_l1.get_output(0), (0, i_center, 0, 0, 0), (1, 1, 128, 144, 176), (1, 1, 1, 1, 1)),
        network.add_slice(feat_l2.get_output(0), (0, i_center, 0, 0, 0), (1, 1, 128, 72, 88), (1, 1, 1, 1, 1)),
        network.add_slice(feat_l3.get_output(0), (0, i_center, 0, 0, 0), (1, 1, 128, 36, 44), (1, 1, 1, 1, 1)),
    ]
    aligned_feat = []
    for i in range(n_frame):
        nbr_feat_l = [
            network.add_slice(feat_l1.get_output(0), (0, i, 0, 0, 0), (1, 1, 128, 144, 176), (1, 1, 1, 1, 1)),
            network.add_slice(feat_l2.get_output(0), (0, i, 0, 0, 0), (1, 1, 128, 72, 88), (1, 1, 1, 1, 1)),
            network.add_slice(feat_l3.get_output(0), (0, i, 0, 0, 0), (1, 1, 128, 36, 44), (1, 1, 1, 1, 1)),
        ]
        aligned_feat.append(pcd_alignment(network, weights, num_feat, nbr_feat_l, ref_feat_l).get_output(0))
    aligned_feat = network.add_concatenation(aligned_feat)
    aligned_feat.axis = 1
    print('aligned_feat', aligned_feat.get_output(0).shape)
    
    feat = tsa(network, weights, num_feat, aligned_feat, n_frame)

    out = make_layer(network, weights, feat, num_feat, 40, 'reconstruction')
    print('out', out.get_output(0).shape)

    out = network.add_convolution(out.get_output(0), num_feat * 4, (3, 3), weights['upconv1.weight'], weights['upconv1.bias'])
    out.stride = (1, 1)
    out.padding = (1, 1)
    print('out', out.get_output(0).shape)

    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (1, 128, 2, 2, 144, 176)
    out.second_transpose = (0, 1, 4, 2, 5, 3)
    print('out', out.get_output(0).shape)
    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (1, 128, 144 * 2, 176 * 2)
    print('out', out.get_output(0).shape)

    out = network.add_activation(out.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    out.alpha = 0.1

    out = network.add_convolution(out.get_output(0), 64 * 4, (3, 3), weights['upconv2.weight'], weights['upconv2.bias'])
    out.stride = (1, 1)
    out.padding = (1, 1)
    print('out', out.get_output(0).shape)

    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (1, 64, 2, 2, 144 * 2, 176 * 2)
    out.second_transpose = (0, 1, 4, 2, 5, 3)
    print('out', out.get_output(0).shape)
    out = network.add_shuffle(out.get_output(0))
    out.reshape_dims = (1, 64, 144 * 2 * 2, 176 * 2 * 2)
    print('out', out.get_output(0).shape)

    out = network.add_activation(out.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    out.alpha = 0.1

    out = network.add_convolution(out.get_output(0), 64, (3, 3), weights['conv_hr.weight'], weights['conv_hr.bias'])
    out.stride = (1, 1)
    out.padding = (1, 1)
    print('out', out.get_output(0).shape)
    out = network.add_activation(out.get_output(0), tensorrt.ActivationType.LEAKY_RELU)
    out.alpha = 0.1

    out = network.add_convolution(out.get_output(0), 3, (3, 3), weights['conv_last.weight'], weights['conv_last.bias'])
    out.stride = (1, 1)
    out.padding = (1, 1)
    print('out', out.get_output(0).shape)

    x_center = network.add_slice(x, (0, i_center, 0, 0, 0), (1, 1, 3, 144, 176), (1, 1, 1, 1, 1))
    x_center = network.add_shuffle(x_center.get_output(0))
    x_center.reshape_dims = (1, 3, 144, 176)
    base = network.add_resize(x_center.get_output(0))
    base.resize_mode = tensorrt.ResizeMode.LINEAR
    base.scales = [1.0, 1.0, 4.0, 4.0]
    print('base', base.get_output(0).shape)

    out = network.add_elementwise(out.get_output(0), base.get_output(0), tensorrt.ElementWiseOperation.SUM)
    print('out', out.get_output(0).shape)

    out.get_output(0).name = 'out'
    network.mark_output(out.get_output(0))

    config = builder.create_builder_config()
    op = builder.create_optimization_profile()
    # op.set_shape('x', (1, 1, 3, 16, 16), (1, 7, 3, 128, 128), (8, 7, 3, 128, 128))
    config.add_optimization_profile(op)
    
    config.max_workspace_size = 1 << 30

    return builder.build_engine(network, config)

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
    #trt = TrtLite(build_engine_dynamic)
    #trt.print_info()
    #trt.save_to_file("edvr.trt")

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
