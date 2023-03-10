_base_ = './yolox-pose_s_8xb32-300e_coco.py'

# model
model = dict(
    train_cfg=dict(
        assigner=dict(
            type='PoseSimOTAAssigner',
            oks_weight=20.0,
            vis_weight=1.0,
            pose_ratio=0.0,
        )))
