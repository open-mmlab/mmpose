#############################################################################
"""You have to fill these args.

_base_ (str): The path to your pretrained model checkpoint.
pretrained_path (str): The path to your pretrained model checkpoint.

interval (int): Interval between pruning two channels. You should ensure you
    can reach your target pruning ratio when the training ends.
normalization_type (str): GroupFisher uses two methods to normlized the channel
    importance, including ['flops','act']. The former uses flops, while the
    latter uses the memory occupation of activation feature maps.
lr_ratio (float): Ratio to decrease lr rate. As pruning progress is unstable,
    you need to decrease the original lr rate until the pruning training work
    steadly without getting nan.

target_flop_ratio (float): The target flop ratio to prune your model.
input_shape (Tuple): input shape to measure the flops.
"""

_base_ = 'mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_aic-coco-256x192.py'  # noqa
pretrained_path = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth'  # noqa

interval = 10
normalization_type = 'act'
lr_ratio = 0.1

target_flop_ratio = 0.51
input_shape = (1, 3, 256, 192)
##############################################################################

architecture = _base_.model

if hasattr(_base_, 'data_preprocessor'):
    architecture.update({'data_preprocessor': _base_.data_preprocessor})
    data_preprocessor = None

architecture.init_cfg = dict(type='Pretrained', checkpoint=pretrained_path)
architecture['_scope_'] = _base_.default_scope

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherAlgorithm',
    architecture=architecture,
    interval=interval,
    mutator=dict(
        type='GroupFisherChannelMutator',
        parse_cfg=dict(type='ChannelAnalyzer', tracer_type='FxTracer'),
        channel_unit_cfg=dict(
            type='GroupFisherChannelUnit',
            default_args=dict(normalization_type=normalization_type, ),
        ),
    ),
)

model_wrapper_cfg = dict(
    type='mmrazor.GroupFisherDDP',
    broadcast_buffers=False,
)

optim_wrapper = dict(
    optimizer=dict(lr=_base_.optim_wrapper.optimizer.lr * lr_ratio))

custom_hooks = getattr(_base_, 'custom_hooks', []) + [
    dict(type='mmrazor.PruningStructureHook'),
    dict(
        type='mmrazor.ResourceInfoHook',
        interval=interval,
        demo_input=dict(
            type='mmrazor.DefaultDemoInput',
            input_shape=input_shape,
        ),
        save_ckpt_thr=[target_flop_ratio],
    ),
]
