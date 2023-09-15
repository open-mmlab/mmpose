_base_ = './td-hm_hrnet-w48_8xb32-210e_deepfashion_full-256x192.py'

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# codec settings
codec = dict(
    type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

model = dict(
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=False))

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(batch_size=32)
