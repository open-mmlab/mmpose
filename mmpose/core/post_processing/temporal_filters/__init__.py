# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_filter
from .gaussian_filter import GaussianFilter
from .one_euro_filter import OneEuroFilter
from .savizky_golay_filter import SavizkyGolayFilter
from .smoothnet import SmoothNet

import os
import torch

__all__ = [
    'build_filter', 'GaussianFilter', 'OneEuroFilter', 'SavizkyGolayFilter','SmoothNet'
]

def get_pretrained_smoothnet(in_length=64, 
    out_length=64, hid_size=512, num_block=3, dropout=0.5, mid_size=256):
    """
    Load pretrained SmoothNet with different input length T.
    """
    model = SmoothNet(
                in_length=in_length, 
                out_length=out_length, 
                hid_size=hid_size, 
                num_block=num_block, 
                dropout=dropout, 
                mid_size=mid_size).cuda()   
    # load pretrained model
    filename = os.path.join('checkpoint/'+'latest_epoch_smoothnet_hm36_t{}.bin'.format(in_length))
    print('Loading checkpoint', filename)
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint.state_dict(), strict=False)
    model.eval()
    return model
