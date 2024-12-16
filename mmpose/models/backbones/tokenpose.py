import torch
from torch import nn
import torch.nn.functional as F
import os
from typing import Sequence
from einops import rearrange, repeat
from mmengine.model import ModuleList, Sequential
from mmengine.model.weight_init import trunc_normal_
import math
from .hrnet import HRNet

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1

class Bottleneck(nn.Module):
    """Bottleneck block for TokenPose stem net.

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    """Transformer Layer for Tokenpose

    Args:
        dim (int): The dimensionality of the input and output feature space for each layer.
        depth (int): The number of transformer layers (or blocks) in the model.
        heads (int): The number of attention heads in the multi-head self-attention mechanism.
        mlp_dim (int): The hidden dimension of the feed-forward network (FFN) in each transformer block.
        dropout (float): The dropout rate applied in both the attention and FFN layers.
        num_keypoints (int, optional): The number of keypoints being processed. If specified, it is used
            to separate keypoints from other positional embeddings in `forward`. Default: None.
        all_attn (bool, optional): Whether to apply positional encoding updates after the first attention
            layer in every block. Default: False.
        scale_with_head (bool, optional): If True, scales the attention scores based on the number of heads.
            Default: False.
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout,num_keypoints=None,all_attn=False, scale_with_head=False):
        super().__init__()
        self.layers = ModuleList()
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        for _ in range(depth):
            self.layers.append(ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None,pos=None):
        outs = []
        for idx,(attn, ff) in enumerate(self.layers):
            if idx>0 and self.all_attn:
                x[:,self.num_keypoints:] += pos
            x = attn(x, mask = mask)
            x = ff(x)
            outs.append(x)
        return outs


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
    
    

@MODELS.register_module()
class TokenPose(BaseBackbone):
    """TokenPose: Learning Keypoint Tokens for Human Pose Estimation.

        Args:
            image_size (list): The size of the input image, given as [height, width].
                Default: [256, 192].
            patch_size (list): The size of each patch, given as [patch_height, patch_width].
                Default: [4, 3].
            num_keypoints (int): The number of keypoints to detect in the pose estimation task.
                Default: 17.
            out_indices (Sequence | int): Output from which stages.
                Defaults to -1, means the last stage.
            dim (int): The embedding dimension of each patch in the Transformer.
                Default: 192.
            depth (int): The number of Transformer layers (blocks).
                Default: 12.
            heads (int): The number of attention heads in the multi-head self-attention mechanism.
                Default: 8.
            mlp_dim (int): The hidden dimension of the feed-forward network (FFN) in each Transformer block.
                Default: 3.
            apply_multi (bool): Whether to use multiple heatmaps for pose estimation.
                Default: True.
            hidden_heatmap_dim (int): The hidden dimension of the heatmap representation.
                Default: `64 * 6`.
            heatmap_dim (int): The output dimension of the heatmap representation.
                Default: `64 * 48`.
            heatmap_size (list): The size of the output heatmap, given as [height, width].
                Default: [64, 48].
            channels (int): The number of channels in the input image (e.g., 3 for RGB images).
                Default: 3.
            dropout (float): The dropout rate used in the Transformer layers.
                Default: 0.0.
            emb_dropout (float): The dropout rate applied to patch embeddings.
                Default: 0.0.
            pos_embedding_type (str): The type of positional embedding used in the Transformer.
                Options include "learnable" (trainable embeddings), "sine", "none", and "sine-full". Default: "learnable".
            frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
                -1 means not freezing any parameters. Defaults to -1.
            size_type(str): Type of size of the model.
                Options include "tiny", "s" (small), "b" (base), and "l" (large). Default: 's'.
            init_cfg (dict, optional): Initialization config dict.
                Defaults to None.
        """
    def __init__(self, *,
                 image_size=[256,192],
                 patch_size=[4,3],
                 num_keypoints=17,
                 out_indices=-1,
                 dim=192,
                 depth=12,
                 heads=8,
                 mlp_dim=3,
                 apply_multi=True,
                 hidden_heatmap_dim=64 * 6,
                 heatmap_dim=64 * 48,
                 heatmap_size=[64, 48],
                 channels=3,
                 dropout=0.,
                 emb_dropout=0.,
                 pos_embedding_type="learnable",
                 frozen_stages=-1,
                 size_type = 's',
                 init_cfg=None):
        super(TokenPose, self).__init__(init_cfg=init_cfg)
        assert isinstance(image_size, list) and isinstance(patch_size, list), 'image_size and patch_size should be list'
        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        assert size_type in ['s', 'l', 'tiny', 'b'], f"Invalid size_type: {size_type}. It must be one of ['s', 'l', 'tiny', 'b']"
        self.patch_size = patch_size
        if size_type == 'tiny':
            num_patches = (image_size[0] // (patch_size[0])) * (image_size[1] // (patch_size[1]))
            h, w = image_size[0] // (self.patch_size[0]), image_size[1] // (self.patch_size[1])
        else:
            num_patches = (image_size[0] // (4 * patch_size[0])) * (image_size[1] // (4 * patch_size[1]))
            h, w = image_size[0] // (4 * self.patch_size[0]), image_size[1] // (4 * self.patch_size[1])
            assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['sine', 'none', 'learnable', 'sine-full']

        self.depth = depth
        self.inplanes = 64
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        self.size_type = size_type

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 2 + self.depth + index #1 for pre-norm, 1 for mlp_head
            assert 0 <= out_indices[i] <= 2 + self.depth, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.layers = ModuleList()
        if self.size_type == 's':
            # stem net
            self.layers.append(Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                self._make_layer(Bottleneck, 64, 4)
            ))
        elif self.size_type == 'b':
            #hrnet32-stage3
            self.pre_feature = HRNetStage3(hrnetbackbone_extra)
            self.layers.append(self.pre_feature)
        elif self.size_type == 'l':
            # hrnet48-stage3
            hrnetbackbone_extra['stage1']['num_channels'] = (64, )
            hrnetbackbone_extra['stage2']['num_channels'] = (48, 96)
            hrnetbackbone_extra['stage3']['num_channels'] = (48, 96, 192)
            hrnetbackbone_extra['stage4']['num_channels'] = (48, 96, 192, 384)
            self.pre_feature = HRNetStage3(hrnetbackbone_extra)
            self.layers.append(self.pre_feature)
        else:
            self.layers.append(Sequential(nn.Identity()))

        # transformer
        if size_type == 'l':
            self.transformer1 = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                            all_attn=self.all_attn, scale_with_head=True)
            self.transformer2 = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                            all_attn=self.all_attn, scale_with_head=True)
            self.transformer3 = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                            all_attn=self.all_attn, scale_with_head=True)
        else:
            self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                           all_attn=self.all_attn)

        self.to_keypoint_token = nn.Identity()

        # mlp_head
        if dim <= hidden_heatmap_dim * 0.5 and apply_multi:
            if self.size_type != 'l':
                self.layers.append(Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden_heatmap_dim),
                    nn.LayerNorm(hidden_heatmap_dim),
                    nn.Linear(hidden_heatmap_dim, heatmap_dim)
                ))
            else:
                self.layers.append(Sequential(
                    nn.LayerNorm(dim * 3),
                    nn.Linear(dim * 3, hidden_heatmap_dim),
                    nn.LayerNorm(hidden_heatmap_dim),
                    nn.Linear(hidden_heatmap_dim, heatmap_dim)
                ))
        else:
            if self.size_type != 'l':
                self.layers.append(Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, heatmap_dim)
                ))
            else:
                self.layers.append(Sequential(
                    nn.LayerNorm(dim * 3),
                    nn.Linear(dim * 3, heatmap_dim)
                ))
        trunc_normal_(self.keypoint_token, std=.02)

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
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
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
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def init_weights(self, *args, **kwargs):
        super(TokenPose, self).init_weights(*args, **kwargs)

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

        for module in self.modules():
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if self.size_type in ['b', 'l']:
            self.pre_feature.init_weights()

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embedding is not None:
            self.pos_embedding.requires_grad = False
        # set dropout to eval model
        self.dropout.eval()
        # freeze patch embedding
        self.patch_to_embedding.eval()
        for param in self.patch_to_embedding.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.layers[0]:
            param.requires_grad = False
        # freeze keypoint_token
        if self.keypoint_token is not None:
            self.keypoint_token.requires_grad = False
        # freeze layers
        if self.frozen_stages <= self.depth:
            if self.size_type == 'l':
                for i in range(1, self.frozen_stages + 1):
                    m1 = self.transformer1.layers[i - 1]
                    m2 = self.transformer2.layers[i - 1]
                    m3 = self.transformer3.layers[i - 1]
                    m1.eval()
                    m2.eval()
                    m3.eval()
                    for param in m1.parameters():
                        param.requires_grad = False
                    for param in m2.parameters():
                        param.requires_grad = False
                    for param in m3.parameters():
                        param.requires_grad = False
            else:
                for i in range(1, self.frozen_stages + 1):
                    m = self.transformer.layers[i - 1]
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
        # freeze the last layer mlp_head
        if self.frozen_stages == 2 + self.depth:
            for param in self.layers[-1]:
                param.requires_grad = False


    def forward(self, img, mask=None):
        p = self.patch_size
        x = img

        outs = []
        x = self.layers[0](x)

        # transformer
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p[0], p2=p[1])
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:  #
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        elif self.pos_embedding_type == "learnable":
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
        x = self.dropout(x)
        outs.append(x)

        if self.size_type == 'l':
            x1 = self.transformer1(x, mask, self.pos_embedding)
            x2 = self.transformer2(x1[-1], mask, self.pos_embedding)
            x3 = self.transformer3(x2[-1], mask, self.pos_embedding)

            x1_out = [self.to_keypoint_token(item[:, 0:self.num_keypoints]) for item in x1]
            x2_out = [self.to_keypoint_token(item[:, 0:self.num_keypoints]) for item in x2]
            x3_out = [self.to_keypoint_token(item[:, 0:self.num_keypoints]) for item in x3]

            for xi_out, xj_out, xk_out in zip(x1_out, x2_out, x3_out):
                x = torch.cat((xi_out, xj_out, xk_out), dim=2)
                outs.append(x)
        else:
            x = self.transformer(x, mask,self.pos_embedding)
            x = [self.to_keypoint_token(item[:, 0:self.num_keypoints]) for item in x]
            outs.extend(x)
            x = x[-1]

        x = self.layers[-1](x)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        outs.append(x)
        outs = [outs[i] for i in self.out_indices]
        return outs


if __name__ == "__main__":
    import torch # 测试输入数据
    img = torch.rand(32, 3, 256, 192) # (batch_size, channels, height, width) #
    # 实例化模型
    model = TokenPose(image_size=[256, 192],
        patch_size=[4,3],
        num_keypoints=37,
        dim=96,
        depth=12,
        heads=8,
        mlp_dim=3,
        apply_init=True,
        heatmap_size=[64, 48],
        channels=256,
        dropout=0.,
        emb_dropout=0.,
        pos_embedding_type="sine-full",
        size_type='s',) # 可选择 's'、'b'、'l' # 打印输入形状 print("Input shape:", img.shape)
    # 调用模型进行推导
    output = model(img)
    print("Output shape:", output.shape)

