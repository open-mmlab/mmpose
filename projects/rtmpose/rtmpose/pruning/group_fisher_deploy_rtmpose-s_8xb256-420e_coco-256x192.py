#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = 'mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'  # noqa
fix_subnet = {
    'backbone.stem.0.conv_(0, 16)_16': 8,
    'backbone.stem.1.conv_(0, 16)_16': 10,
    'backbone.stem.2.conv_(0, 32)_32': 11,
    'backbone.stage1.0.conv_(0, 64)_64': 32,
    'backbone.stage1.1.short_conv.conv_(0, 32)_32': 32,
    'backbone.stage1.1.main_conv.conv_(0, 32)_32': 23,
    'backbone.stage1.1.blocks.0.conv1.conv_(0, 32)_32': 25,
    'backbone.stage1.1.final_conv.conv_(0, 64)_64': 25,
    'backbone.stage2.0.conv_(0, 128)_128': 71,
    'backbone.stage2.1.short_conv.conv_(0, 64)_64': 61,
    'backbone.stage2.1.main_conv.conv_(0, 64)_64': 62,
    'backbone.stage2.1.blocks.0.conv1.conv_(0, 64)_64': 57,
    'backbone.stage2.1.blocks.1.conv1.conv_(0, 64)_64': 59,
    'backbone.stage2.1.final_conv.conv_(0, 128)_128': 69,
    'backbone.stage3.0.conv_(0, 256)_256': 177,
    'backbone.stage3.1.short_conv.conv_(0, 128)_128': 122,
    'backbone.stage3.1.main_conv.conv_(0, 128)_128': 123,
    'backbone.stage3.1.blocks.0.conv1.conv_(0, 128)_128': 125,
    'backbone.stage3.1.blocks.1.conv1.conv_(0, 128)_128': 123,
    'backbone.stage3.1.final_conv.conv_(0, 256)_256': 171,
    'backbone.stage4.0.conv_(0, 512)_512': 351,
    'backbone.stage4.1.conv1.conv_(0, 256)_256': 256,
    'backbone.stage4.1.conv2.conv_(0, 512)_512': 367,
    'backbone.stage4.2.short_conv.conv_(0, 256)_256': 183,
    'backbone.stage4.2.main_conv.conv_(0, 256)_256': 216,
    'backbone.stage4.2.blocks.0.conv1.conv_(0, 256)_256': 238,
    'backbone.stage4.2.blocks.0.conv2.pointwise_conv.conv_(0, 256)_256': 195,
    'backbone.stage4.2.final_conv.conv_(0, 512)_512': 187
}
divisor = 16
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
