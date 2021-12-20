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
            synchronous=False,
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            result_buffer='pose_result',
            output_buffer='frame',
            has_det_results=True),
        # 'MatchStickMenNode':
        # This node highlights human keypoints and skeletons in the image.
        # It can also launch a dynamically expanding red heart from the
        # middle of hands if the person is posing a "hand heart" gesture.
        # Pose results is needed.
        dict(
            type='MatchStickMenNode',
            name='Visualizer',
            enable_key='n',
            frame_buffer='frame',
            output_buffer='vis_matchstickmen',
            background_color=(0, 0, 0),
        ),
        # 'ELkHornNode':
        # This node add an elkhorn on top of the person in the image.
        # Pose results is needed.
        dict(
            type='ELkHornNode',
            name='Visualizer',
            enable_key='c',
            frame_buffer='vis_matchstickmen',
            output_buffer='vis_elk_horn'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='Helper',
            enable_key='h',
            enable=True,
            frame_buffer='vis_elk_horn',
            output_buffer='vis',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"h": Show help information', '"n": MatchStickMen Effect',
                '"c": ELkHorn Effect', '"m": Show diagnostic information',
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
            frame_buffer='vis',
            output_buffer='_display_'
        ),  # `_frame_` is a runner-reserved buffer
        # 'RecorderNode':
        # This node record the frames into a local file. It can save the
        # visualiztion results. Uncommit the following lines to turn it on.
        # dict(
        #     type='RecorderNode',
        #     name='Recorder',
        #     out_video_file='webcam_matchstickman.mp4',
        #     frame_buffer='display',
        #     output_buffer='_display_')
    ])
