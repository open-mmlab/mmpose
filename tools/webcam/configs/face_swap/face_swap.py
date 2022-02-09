# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    name='FaceSwap',
    camera_id=0,
    camera_fps=20,
    synchronous=False,
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
            device='cpu',
            input_buffer='_input_',  # `_input_` is a runner-reserved buffer
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
            device='cpu',
            cls_names=['person'],
            input_buffer='det_result',
            output_buffer='pose_result'),
        dict(
            type='ModelResultBindingNode',
            name='ResultBinder',
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            result_buffer='pose_result',
            output_buffer='frame'),
        dict(
            type='FaceSwapNode',
            name='FaceSwapper',
            mode_key='s',
            frame_buffer='frame',
            output_buffer='face_swap'),
        dict(
            type='PoseVisualizerNode',
            name='Visualizer',
            enable_key='v',
            frame_buffer='face_swap',
            output_buffer='vis_pose'),
        dict(
            type='NoticeBoardNode',
            name='Help Information',
            enable_key='h',
            content_lines=[
                'Swap your faces! ',
                'Hot-keys:',
                '"v": Toggle the pose visualization on/off.',
                '"s": Switch between modes: Shuffle, Clone and None',
                '"h": Show help information',
                '"m": Show diagnostic information',
                '"q": Exit',
            ],
            frame_buffer='vis_pose',
            output_buffer='vis_notice'),
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
            out_video_file='faceswap_output.mp4',
            frame_buffer='display',
            output_buffer='_display_')
    ])
