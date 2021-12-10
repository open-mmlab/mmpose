# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    name='Human Pose and Effects',
    camera_id=0,
    camera_fps=30,
    display_delay=0,
    frame_buffer_size=20,
    user_buffers=[
        ('det_result', 2),
        ('pose_result', 2),
        ('frame', 2),
        ('vis_pose', 2),
        ('vis_sunglasses', 2),
        ('vis_bugeye', 2),
    ],
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
            input_buffer='_input_',
            output_buffer='det_result'),
        dict(
            type='TopDownPoseEstimatorNode',
            name='TopDown Pose Estimator',
            model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
            'topdown_heatmap/coco-wholebody/'
            'vipnas_res50_coco_wholebody_256x192_dark.py',
            model_checkpoint='https://openmmlab-share.oss-cn-hangzhou'
            '.aliyuncs.com/mmpose/top_down/vipnas/'
            'vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth',
            cls_names=['person'],
            input_buffer='det_result',
            output_buffer='pose_result'),
        dict(
            type='ModelResultBindingNode',
            name='ResultBinder',
            frame_buffer='_frame_',
            result_buffer='pose_result',
            output_buffer='frame'),
        dict(
            type='PoseVisualizerNode',
            name='Visualizer',
            enable_key='v',
            frame_buffer='frame',
            output_buffer='vis_pose'),
        # dict(
        #     type='SunglassesNode',
        #     name='Visualizer',
        #     enable_key='s',
        #     frame_buffer='vis_pose',
        #     output_buffer='vis_sunglasses'),
        # dict(
        #     type='BugEyeNode',
        #     name='Visualizer',
        #     enable_key='b',
        #     frame_buffer='vis_sunglasses',
        #     output_buffer='vis_bugeye'),
        dict(
            type='MonitorNode',
            name='Monitor',
            enable_key='m',
            style='fancy',
            input_buffer='vis_pose',
            output_buffer='_display_')
    ])
