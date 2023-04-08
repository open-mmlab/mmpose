# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    # Basic configurations of the executor
    name='Pose Estimation',
    camera_id=0,
    # Define nodes.
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
            name='detector',
            model_config='projects/rtmpose/rtmdet/person/'
            'rtmdet_nano_320-8xb32_coco-person.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/v1/'
            'projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',  # noqa
            input_buffer='_input_',  # `_input_` is an executor-reserved buffer
            output_buffer='det_result'),
        # 'TopdownPoseEstimatorNode':
        # This node performs keypoint detection from the frame image using an
        # MMPose top-down model. Detection results is needed.
        dict(
            type='TopdownPoseEstimatorNode',
            name='human pose estimator',
            model_config='projects/rtmpose/rtmpose/body_2d_keypoint/'
            'rtmpose-t_8xb256-420e_coco-256x192.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/v1/'
            'projects/rtmpose/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth',  # noqa
            labels=['person'],
            input_buffer='det_result',
            output_buffer='human_pose'),
        # 'ObjectAssignerNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='human_pose',
            output_buffer='frame'),
        # 'ObjectVisualizerNode':
        # This node draw the pose visualization result in the frame image.
        # Pose results is needed.
        dict(
            type='ObjectVisualizerNode',
            name='object visualizer',
            enable_key='v',
            enable=True,
            show_bbox=True,
            must_have_keypoint=False,
            show_keypoint=True,
            input_buffer='frame',
            output_buffer='vis'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='instruction',
            enable_key='h',
            enable=True,
            input_buffer='vis',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
            ],
        ),
        # 'MonitorNode':
        # This node show diagnostic information in the frame image. It can
        # be used for debugging or monitoring system resource status.
        dict(
            type='MonitorNode',
            name='monitor',
            enable_key='m',
            enable=False,
            input_buffer='vis_notice',
            output_buffer='display'),
        # 'RecorderNode':
        # This node save the output video into a file.
        dict(
            type='RecorderNode',
            name='recorder',
            out_video_file='webcam_api_demo.mp4',
            input_buffer='display',
            output_buffer='_display_'
            # `_display_` is an executor-reserved buffer
        )
    ])
