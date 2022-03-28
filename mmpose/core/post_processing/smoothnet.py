
'''
source from smoothnet: 
arxiv.org/pdf/2112.13715.pdf
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from .builder import FILTERS


class Encoder(nn.Module):
    def __init__(self, in_length, hid_size):
        super(Encoder, self).__init__()
        self.conv1d = nn.Linear(in_features=in_length, out_features=hid_size, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, input):
        return self.lrelu(self.conv1d(input))

class Dense(nn.Module):
    '''
        N Residual Block
    '''
    def __init__(self, hid_size=256, mid_size=256, dropout=0.25):
        super(Dense, self).__init__()
        self.linear_1 = nn.Linear(in_features=hid_size, out_features=mid_size, bias=True)
        self.linear_2 = nn.Linear(in_features=mid_size, out_features=hid_size, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, input):
        x = input
        x_f = self.lrelu(self.dropout(self.linear_1(input)))
        return self.lrelu(self.dropout(self.linear_2(x_f))) + x

class Decoder(nn.Module):
    def __init__(self, hid_size, out_length):
        super(Decoder, self).__init__()
        self.conv1d = nn.Linear(in_features=hid_size, out_features=out_length, bias=True)

    def forward(self, input):
        return self.conv1d(input)

@FILTERS.register_module(name=['SmoothNet', 'smoothnet'])
class SmoothNet(nn.Module):
    """
    SmoothNet is a plug-and-play temporal-only network to refine human poses.
    Smoothing the outputs of existing 2d/3d/6d pose backbones.
    Args:
        in_length (int):
                    The length of the input window size
                    (64 frames by default).
        out_length (int):
                    The length of the output window size
                    (the same as in_length by default).       
        hid_size (int):
                    Hidden size
        num_block (int):
                    N blocks of the middle dense layers
        dropout (float)
        mid_size (int): 
                    Another hidden size 
                    (256 by default).       
                
    Returns:
        smoothed poses (np.ndarray, torch.tensor)
    """
    def __init__(self, 
                in_length=64, 
                out_length=64, 
                hid_size=512, 
                num_block=3, 
                dropout=0.5, 
                mid_size=256):
        super(SmoothNet, self).__init__()

        self.enc = Encoder(in_length, hid_size)
        self.dec = Decoder(hid_size, out_length)

        ResidualBlock = []
        for i in range(num_block):
            ResidualBlock.append(Dense(hid_size=hid_size, mid_size=mid_size, dropout=dropout))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

    def forward(self, x):
        T, K, C = x.size()

        assert len(x.shape) == 3
        assert T == in_length == out_length
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x
            else:
                x = x.cuda()
        elif isinstance(x, np.array):
            x = torch.from_numpy(x).cuda()

        x = x.reshape(1, T, K*C) #[N=1, T, K*C]
        x = x.permute(0,2,1) #[N=1, K*C, T]

        # smooth all axes parallelly
        x_ = self.enc(x)
        x_ = self.ResidualBlock(x_)
        smooth_poses = self.dec(x_)
        
        # return the original shape
        smooth_poses = smooth_poses.permute(0,2,1)
        smooth_poses = smooth_poses.reshape(N,T,K,-1)

        if isinstance(x_type, torch.Tensor):
            if x_type.is_cuda:
                smooth_poses = smooth_poses
            else:
                smooth_poses = smooth_poses.cpu()
        elif isinstance(x_type, np.array):
            smooth_poses = smooth_poses.cpu().numpy()
        return smooth_poses


if __name__ == "__main__":
    # load model:
    model = SmoothNet(
                in_length=64, 
                out_length=64, 
                hid_size=512, 
                num_block=3, 
                dropout=0.5, 
                mid_size=256)

    # load pretrain model
    filename = os.path.join(args.checkpoint)
    print('Loading checkpoint', filename)
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict() 
    state_dict = {k:v for k, v in checkpoint['model_pos'].items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # test directly with smoothnet
    with torch.no_grad():
            model.eval()

