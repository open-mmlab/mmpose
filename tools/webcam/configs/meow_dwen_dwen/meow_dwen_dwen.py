# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    # Basic configurations of the runner
    name='Little fans of 2022 Beijing Winter Olympics',
    # Cat image
    camera_id='https://user-images.githubusercontent.com/'
    '15977946/152932036-b5554cf8-24cf-40d6-a358-35a106013f11.jpeg',
    # Dog image
    # camera_id='https://user-images.githubusercontent.com/'
    # '15977946/152932051-cd280b35-8066-45a0-8f52-657c8631aaba.jpg',
    camera_fps=20,
    nodes=[
        dict(
            type='DetectorNode',
            name='Detector',
            model_config='demo/mmdetection_cfg/'
            'ssdlite_mobilenetv2_scratch_600e_coco.py',
            model_checkpoint='https://download.openmmlab.com'
            '/mmdetection/v2.0/ssd/'
            'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
            'scratch_600e_coco_20210629_110627-974d9307.pth',
            input_buffer='_input_',  # `_input_` is a runner-reserved buffer
            output_buffer='det_result'),
        dict(
            type='TopDownPoseEstimatorNode',
            name='Animal Pose Estimator',
            model_config='configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap'
            '/ap10k/hrnet_w32_ap10k_256x256.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/animal/'
            'hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth',
            cls_names=['cat', 'dog'],
            input_buffer='det_result',
            output_buffer='animal_pose'),
        dict(
            type='TopDownPoseEstimatorNode',
            name='TopDown Pose Estimator',
            model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
            'topdown_heatmap/coco-wholebody/'
            'vipnas_res50_coco_wholebody_256x192_dark.py',
            model_checkpoint='https://openmmlab-share.oss-cn-hangzhou'
            '.aliyuncs.com/mmpose/top_down/vipnas/'
            'vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth',
            device='cpu',
            cls_names=['person'],
            input_buffer='animal_pose',
            output_buffer='human_pose'),
        dict(
            type='ModelResultBindingNode',
            name='ResultBinder',
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            result_buffer='human_pose',
            output_buffer='frame'),
        dict(
            type='XDwenDwenNode',
            name='XDwenDwen',
            mode_key='s',
            resource_file='tools/webcam/configs/meow_dwen_dwen/'
            'resource-info.json',
            out_shape=(480, 480),
            frame_buffer='frame',
            output_buffer='vis'),
        dict(
            type='NoticeBoardNode',
            name='Helper',
            enable_key='h',
            enable=False,
            frame_buffer='vis',
            output_buffer='vis_notice',
            content_lines=[
                'Let your pet put on a costume of Bing-Dwen-Dwen, '
                'the mascot of 2022 Beijing Winter Olympics. Have fun!', '',
                'Hot-keys:', '"s": Change the background',
                '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
            ],
        ),
        dict(
            type='MonitorNode',
            name='Monitor',
            enable_key='m',
            enable=False,
            frame_buffer='vis_notice',
            output_buffer='display'),
        dict(
            type='RecorderNode',
            name='Recorder',
            out_video_file='record.mp4',
            frame_buffer='display',
            output_buffer='_display_'
            # `_display_` is a runner-reserved buffer
        )
    ])
