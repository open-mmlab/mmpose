from torch import nn, Tensor
from typing import Optional
import torch
import copy
import torch.nn.functional as F
import os
from typing import Sequence
import math

from mmengine.model import ModuleList, Sequential
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .hrnet import HRNet

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for TransPose-R pre-norm.

        Args:
            inplanes (int): The number of input channels for the first convolutional layer.
            planes (int): The number of output channels for the intermediate convolutional layers.
            stride (int, optional): The stride for the second convolutional layer.
                Used to downsample the spatial resolution. Default: 1.
            downsample (nn.Module, optional): A downsampling layer applied to the residual connection
                if the input and output dimensions do not match. Default: None.
                expansion (int, optional): The ratio of `out_channels / mid_channels`, where
            expansion (int, optional): The ratio of `out_channels / mid_channels`, where
                `mid_channels` corresponds to the input and output channels of `conv2`.
                This determines how much the number of channels is expanded in the final
                convolutional layer (`conv3`). Default: 4.
        """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


hrnetbackbone_extra = dict(  # hrnetbackbone_extra(dict): TokenPose_Base and TokenPose_Large backbone configuration.
    stage1=dict(
        num_modules=1,
        num_branches=1,
        block='BOTTLENECK',
        num_blocks=(4,),
        num_channels=(64,)
    ),
    stage2=dict(
        num_modules=1,
        num_branches=2,
        block='BASIC',
        num_blocks=(4, 4),
        num_channels=(32, 64)
    ),
    stage3=dict(
        num_modules=4,
        num_branches=3,
        block='BASIC',
        num_blocks=(4, 4, 4),
        num_channels=(32, 64, 128)
    ),
    stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(32, 64, 128, 256)
        )
)
class HRNetStage3(HRNet):
    def __init__(self, *args, **kwargs):
        super(HRNetStage3, self).__init__(*args, **kwargs)

    def forward(self, img):
        """Forward function."""
        x = self.conv1(img)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        return y_list[0]



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        outputs = []
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)
            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            outputs = [self.norm(o) for o in outputs]

        if self.return_atten_map:
            return outputs, torch.stack(atten_maps_list)
        else:
            return outputs


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


@MODELS.register_module()
class TransPose(BaseBackbone):
    def __init__(self, *,
                 image_size=[192, 256],
                 deconv_with_bias=False,
                 dim = 256,
                 dim_feedforward = 1024,
                 encoder_layers_num = 3,
                 num_head = 8,
                 pos_embedding_type = 'sine',
                 num_deconv_layers = 1,
                 num_deconv_filters = [256],
                 num_deconv_kernels = [4],
                 final_conv_kernel = 1,
                 layers = [3, 4, 6, 3],
                 num_joints = 37,
                 out_indices=-1,
                 frozen_stages=-1,
                 type_cnn = 'resnet',
                 type_hrnet = 'w32',
                 init_cfg=None):
        super(TransPose, self).__init__(init_cfg=init_cfg)
        assert pos_embedding_type in ['sine', 'none', 'learnable', 'sine-full']
        assert type_cnn in ['resnet', 'hrnet'], f"Invalid type_cnn: {type_cnn}. It must be one of ['resnet', 'hrnet']"
        assert type_hrnet in ['w32', 'w48', None], f"Invalid type_cnn: {type_hrnet}. It must be one of ['w32', 'w48', None]"
        self.inplanes = 64
        self.deconv_with_bias = deconv_with_bias
        self.type_cnn = type_cnn
        self.type_hrnet = type_hrnet

        self.d_model = dim
        self.dim_feedforward = dim_feedforward
        self.encoder_layers_num = encoder_layers_num
        self.n_head = num_head
        self.pos_embedding_type = pos_embedding_type
        self.w, self.h = image_size[0], image_size[1]
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels
        self.final_conv_kernel = final_conv_kernel
        self.num_joints = num_joints

        self.layers = ModuleList()
        if self.type_cnn == 'resnet':
            #backbone for resnet
            self.layers.append(Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False),
                nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_layer(Bottleneck, 64, layers[0]),
                self._make_layer(Bottleneck, 128, layers[1], stride=2),
                nn.Conv2d(self.inplanes, self.d_model, 1, bias=False)
            ))
        else:
            # backbone for hrnet
            if self.type_hrnet == 'w32':
                # hrnet32-stage3
                self.pre_feature = HRNetStage3(hrnetbackbone_extra)
                self.layers.append(Sequential(self.pre_feature,
                                              nn.Conv2d(32, self.d_model, 1, bias=False)))
            else:
                # hrnet48-stage3
                hrnetbackbone_extra['stage1']['num_channels'] = (64,)
                hrnetbackbone_extra['stage2']['num_channels'] = (48, 96)
                hrnetbackbone_extra['stage3']['num_channels'] = (48, 96, 192)
                hrnetbackbone_extra['stage4']['num_channels'] = (48, 96, 192, 384)
                self.pre_feature = HRNetStage3(hrnetbackbone_extra)
                self.layers.append(Sequential(self.pre_feature,
                                              nn.Conv2d(48, self.d_model, 1, bias=False)))

        self._make_position_embedding(self.w, self.h, self.d_model, pos_embedding_type)

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            self.encoder_layers_num,
            return_atten_map=False
        )

        if self.type_cnn == 'resnet':
            # used for deconv layers
            self.inplanes = self.d_model
            self.deconv_layers = self._make_deconv_layer(
                self.num_deconv_layers,  # 1
                self.num_deconv_filters,  # [d_model]
                self.num_deconv_kernels,  # [4]
            )
            self.layers.append(self.deconv_layers)
        else:
            self.layers.append(Sequential(nn.Identity()))

        self.final_layer = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.num_joints,
            kernel_size=self.final_conv_kernel,
            stride=1,
            padding=1 if self.final_conv_kernel == 3 else 0
        )
        self.layers.append(self.final_layer)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 3 + self.encoder_layers_num + index #1 for pre-norm, 1 for head, 1 for deconv_layers
            assert 0 <= out_indices[i] <= 3 + self.encoder_layers_num, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                if self.type_cnn == 'resnet':
                    self.pe_h = h // 8
                    self.pe_w = w // 8
                else:
                    self.pe_h = h // 4
                    self.pe_w = w // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        x = self.layers[0](x)
        outs.append(x)
        
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        outs.extend(x)
        x = x[-1]
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        x = self.layers[-2](x)
        outs.append(x)
        x = self.layers[-1](x)
        outs.append(x)

        outs = [outs[i] for i in self.out_indices]
        return outs

    def init_weights(self):
        super(TransPose, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)

        if self.type_cnn == 'hrnet':
            self.pre_feature.init_weights()

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embedding is not None:
            self.pos_embedding.requires_grad = False
        # freeze pre-norm
        for param in self.layers[0]:
            param.requires_grad = False
        # freeze layers
        if self.frozen_stages <= self.encoder_layers_num:
            for i in range(1, self.frozen_stages + 1):
                m = self.global_encoder.layers[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        # freeze the last 2 layer
        if self.frozen_stages == 2 + self.encoder_layers_num:
            for param in self.layers[-2]:
                param.requires_grad = False
        if self.frozen_stages == 3 + self.encoder_layers_num:
            for param in self.layers[-1]:
                param.requires_grad = False