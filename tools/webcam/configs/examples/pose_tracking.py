# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    # Basic configurations of the runner
    name='Pose Estimation',
    camera_id=0,
    camera_fps=20,
    synchronous=False,
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
        dict(
            type='PoseTrackerNode',
            name='PoseTracker',
            det_model_config='demo/mmdetection_cfg/'
            'ssdlite_mobilenetv2_scratch_600e_coco.py',
            det_model_checkpoint='https://download.openmmlab.com'
            '/mmdetection/v2.0/ssd/'
            'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
            'scratch_600e_coco_20210629_110627-974d9307.pth',
            pose_model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
            'topdown_heatmap/coco-wholebody/'
            'vipnas_mbv3_coco_wholebody_256x192_dark.py',
            pose_model_checkpoint='https://download.openmmlab.com/mmpose/'
            'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
            '-e2158108_20211205.pth',
            det_interval=10,
            cls_names=['person'],
            smooth=True,
            device='cuda:0',
            input_buffer='_input_',  # `_input_` is a runner-reserved buffer
            output_buffer='human_pose'),
        # 'ModelResultBindingNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ModelResultBindingNode',
            name='ResultBinder',
            frame_buffer='_frame_',  # `_frame_` is a runner-reserved buffer
            result_buffer='human_pose',
            output_buffer='frame'),
        # 'PoseVisualizerNode':
        # This node draw the pose visualization result in the frame image.
        # Pose results is needed.
        dict(
            type='PoseVisualizerNode',
            name='Visualizer',
            enable_key='v',
            frame_buffer='frame',
            output_buffer='vis'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='Helper',
            enable_key='h',
            enable=True,
            frame_buffer='vis',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"s": Sunglasses effect B-)', '"b": Bug-eye effect 0_0',
                '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
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
            output_buffer='display'),
        # 'RecorderNode':
        # This node save the output video into a file.
        dict(
            type='RecorderNode',
            name='Recorder',
            out_video_file='record.mp4',
            frame_buffer='display',
            output_buffer='_display_'
            # `_display_` is a runner-reserved buffer
        )
    ])
