# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    # Basic configurations of the runner
    name='Human Pose and Effects',
    camera_id=0,
    camera_fps=30,

    # Define nodes.
    #
    # The configuration of a node usually includes:
    #   1. 'type': Node class name
    #   2. 'name': Node name
    #   3. I/O buffers (e.g. 'input_buffer', 'output_buffer'): specify the
    #       input and output buffer names. This may depend on the node class.
    #   4. 'enable_key': assign a hot-key to toggle enable/disable this node.
    #       This may depend on the node class.
    #   5. Other class-specific arguments
    nodes=[
        # 'DetectorNode':
        # This node performs object detection from the frame image using an
        # MMDetection model.
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
        # 'TopDownPoseEstimatorNode':
        # This node performs keypoint detection from the frame image using an
        # MMPose top-down model. Detection results is needed.
        dict(
            type='TopDownPoseEstimatorNode',
            name='Human Pose Estimator',
            model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
            'topdown_heatmap/coco-wholebody/'
            'vipnas_mbv3_coco_wholebody_256x192_dark.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/top_down/'
            'vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
            '-e2158108_20211205.pth',
            cls_names=['person'],
            input_buffer='det_result',
            output_buffer='pose_result'),
        # 'ModelResultBindingNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ModelResultBindingNode',
            name='ResultBinder',
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            result_buffer='pose_result',
            output_buffer='frame'),
        # 'PoseVisualizerNode':
        # This node draw the pose visualization result in the frame image.
        # Pose results is needed.
        dict(
            type='PoseVisualizerNode',
            name='Visualizer',
            enable_key='v',
            enable=False,
            frame_buffer='frame',
            output_buffer='vis'),
        # 'ValentineMagicNode':
        # This node draw heart in the image.
        # It can launch dynamically expanding heart from the middle of
        # hands if the persons pose a "hand heart" gesture or blow a kiss.
        # Only there are two persons in the image can trigger this effect.
        # Pose results is needed.
        dict(
            type='ValentineMagicNode',
            name='Visualizer',
            enable_key='l',
            frame_buffer='vis',
            output_buffer='vis_heart',
        ),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='Helper',
            enable_key='h',
            enable=False,
            frame_buffer='vis_heart',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"h": Show help information', '"l": LoveHeart Effect',
                '"v": PoseVisualizer', '"m": Show diagnostic information',
                '"q": Exit'
            ],
        ),
        # 'MonitorNode':
        # This node show diagnostic information in the frame image. It can
        # be used for debugging or monitoring system resource status.
        dict(
            type='MonitorNode',
            name='Monitor',
            enable_key='m',
            enable=False,
            frame_buffer='vis_notice',
            output_buffer='display'),  # `_frame_` is a runner-reserved buffer
        # 'RecorderNode':
        # This node record the frames into a local file. It can save the
        # visualiztion results. Uncommit the following lines to turn it on.
        dict(
            type='RecorderNode',
            name='Recorder',
            out_video_file='record.mp4',
            frame_buffer='display',
            output_buffer='_display_')
    ])
