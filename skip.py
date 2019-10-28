"""
Script to generate encoder-decoder models. Same as from the original DIP codebase: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/models/skip.py, with a few additional lines for DIP-linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True,use_bn=True,stride=2,use_fc=False,input_shape=()):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    
    if use_fc:
        assert (len(input_shape) > 0)
        input_size = np.prod(input_shape)
        img_size = int(input_size)
    
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
       
        if use_bn:
            if not use_fc:
                model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            if use_fc:
                skip.add(Flatten())
            else:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            if use_bn:
                skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
            
        if use_fc:
            deeper.add(Flatten())
            deeper.add(nn.Linear(input_size, num_channels_down[i]))
        else:
            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], stride, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        if use_bn:
            if not use_fc:
                deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        if use_fc:
            deeper.add(Flatten())
            deeper.add(nn.Linear(num_channels_down[i], num_channels_down[i]))
        else:
            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        if use_bn:
            if not use_fc:
                deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        if stride > 1:
            if not use_fc:
                deeper.add(nn.Upsample(scale_factor=stride, mode=upsample_mode[i]))

        if use_fc:
            model_tmp.add(Flatten())
            model_tmp.add(nn.Linear(num_channels_skip[i] + k, num_channels_up[i]))
        else:
            model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        if use_bn:
            if not use_fc:
                model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            if use_fc:
                model_tmp.add(Flatten())
                model_tmp.add(nn.Linear(num_channels_up[i], num_channels_up[i]))
            else:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            if use_bn:
                if not use_fc:
                    model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        input_size = num_channels_down[i]
        model_tmp = deeper_main

    if use_fc:
        model.add(Flatten())
        model.add(nn.Linear(num_channels_up[0], img_size))
    else:
        model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    if use_fc:
        model.add(ReshapeToImg(input_shape))

    return model


