from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.runner import LogProcessor, TestLoop, ValLoop
from mmengine.visualization import LocalVisBackend

from mmpose.engine.hooks import PoseVisualizationHook
from mmpose.visualization import PoseLocalVisualizer

default_scope = None
# hooks
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=PoseVisualizationHook, enable=False),
)

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [dict(type=LocalVisBackend)]
visualizer = dict(
    type=PoseLocalVisualizer, vis_backends=vis_backends, name='visualizer')

# logger
log_processor = dict(
    type=LogProcessor, window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend='local')

# training/validation/testing progress
train_cfg = dict()
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
