#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = 'mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py'  # noqa
fix_subnet = {
    'backbone.stem.0.conv_(0, 16)_16': 8,
    'backbone.stem.1.conv_(0, 16)_16': 9,
    'backbone.stem.2.conv_(0, 32)_32': 9,
    'backbone.stage1.0.conv_(0, 64)_64': 32,
    'backbone.stage1.1.short_conv.conv_(0, 32)_32': 30,
    'backbone.stage1.1.main_conv.conv_(0, 32)_32': 29,
    'backbone.stage1.1.blocks.0.conv1.conv_(0, 32)_32': 24,
    'backbone.stage1.1.final_conv.conv_(0, 64)_64': 27,
    'backbone.stage2.0.conv_(0, 128)_128': 62,
    'backbone.stage2.1.short_conv.conv_(0, 64)_64': 63,
    'backbone.stage2.1.main_conv.conv_(0, 64)_64': 64,
    'backbone.stage2.1.blocks.0.conv1.conv_(0, 64)_64': 56,
    'backbone.stage2.1.blocks.1.conv1.conv_(0, 64)_64': 62,
    'backbone.stage2.1.final_conv.conv_(0, 128)_128': 65,
    'backbone.stage3.0.conv_(0, 256)_256': 167,
    'backbone.stage3.1.short_conv.conv_(0, 128)_128': 127,
    'backbone.stage3.1.main_conv.conv_(0, 128)_128': 128,
    'backbone.stage3.1.blocks.0.conv1.conv_(0, 128)_128': 124,
    'backbone.stage3.1.blocks.1.conv1.conv_(0, 128)_128': 123,
    'backbone.stage3.1.final_conv.conv_(0, 256)_256': 172,
    'backbone.stage4.0.conv_(0, 512)_512': 337,
    'backbone.stage4.1.conv1.conv_(0, 256)_256': 256,
    'backbone.stage4.1.conv2.conv_(0, 512)_512': 379,
    'backbone.stage4.2.short_conv.conv_(0, 256)_256': 188,
    'backbone.stage4.2.main_conv.conv_(0, 256)_256': 227,
    'backbone.stage4.2.blocks.0.conv1.conv_(0, 256)_256': 238,
    'backbone.stage4.2.blocks.0.conv2.pointwise_conv.conv_(0, 256)_256': 195,
    'backbone.stage4.2.final_conv.conv_(0, 512)_512': 163
}
divisor = 8
##############################################################################

architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherDeploySubModel',
    architecture=architecture,
    fix_subnet=fix_subnet,
    divisor=divisor,
)
