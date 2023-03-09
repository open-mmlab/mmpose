# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    name='Gesture Recognition',
    camera_id=0,
    camera_max_fps=15,
    synchronous=False,
    buffer_sizes=dict(_input_=20, det_result=10),
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
            model_config='demo/mmdetection_cfg/'
            'ssdlite_mobilenetv2_scratch_600e_onehand.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/'
            'mmdet_pretrained/'
            'ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth',
            input_buffer='_input_',
            output_buffer='det_result',
            multi_input=True),
        # 'HandGestureRecognizerNode':
        # This node performs gesture recognition from the video clip using an
        # MMPose gesture recognition model. Hand detection results is needed.
        dict(
            type='HandGestureRecognizerNode',
            name='gesture recognizer',
            model_config='configs/hand/gesture_sview_rgbd_vid/mtut/'
            'nvgesture/i3d_nvgesture_bbox_112x112_fps15_rgb.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/'
            'gesture/mtut/'
            'i3d_nvgesture_bbox_112x112_fps15-363b5956_20220530.pth',
            input_buffer='det_result',
            output_buffer='gesture',
            fps=15,
            multi_input=True),
        # 'ObjectAssignerNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='gesture',
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
            show_keypoint=False,
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
                'This is a demo for gesture visualization. '
                'Have fun!', '', 'Hot-keys:',
                '"v": Hand bbox & Gesture visualization',
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
            out_video_file='gesture.mp4',
            input_buffer='display',
            output_buffer='_display_'
            # `_display_` is an executor-reserved buffer
        )
    ])
