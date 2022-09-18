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
            model_config='demo/mmdetection_cfg/'
            'ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py',
            model_checkpoint='https://download.openmmlab.com'
            '/mmdetection/v2.0/ssd/'
            'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
            'scratch_600e_coco_20210629_110627-974d9307.pth',
            input_buffer='_input_',  # `_input_` is an executor-reserved buffer
            output_buffer='det_result'),
        # 'TopDownPoseEstimatorNode':
        # This node performs keypoint detection from the frame image using an
        # MMPose top-down model. Detection results is needed.
        dict(
            type='TopDownPoseEstimatorNode',
            name='human pose estimator',
            model_config='configs/wholebody_2d_keypoint/'
            'topdown_heatmap/coco-wholebody/'
            'td-hm_vipnas-mbv3_dark-8xb64-210e_coco-wholebody-256x192.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/'
            'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
            '-e2158108_20211205.pth',
            labels=['person'],
            input_buffer='det_result',
            output_buffer='human_pose'),
        dict(
            type='TopDownPoseEstimatorNode',
            name='animal pose estimator',
            model_config='configs/animal_2d_keypoint/topdown_heatmap/'
            'animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/animal/'
            'hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
            labels=['cat', 'dog', 'horse', 'sheep', 'cow'],
            input_buffer='human_pose',
            output_buffer='animal_pose'),
        # 'ObjectAssignerNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='animal_pose',
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
        # 'SunglassesNode':
        # This node draw the sunglasses effect in the frame image.
        # Pose results is needed.
        dict(
            type='SunglassesEffectNode',
            name='sunglasses',
            enable_key='s',
            enable=False,
            input_buffer='vis',
            output_buffer='vis_sunglasses'),
        # # 'BigeyeEffectNode':
        # # This node draw the big-eye effetc in the frame image.
        # # Pose results is needed.
        dict(
            type='BigeyeEffectNode',
            name='big-eye',
            enable_key='b',
            enable=False,
            input_buffer='vis_sunglasses',
            output_buffer='vis_bigeye'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='instruction',
            enable_key='h',
            enable=True,
            input_buffer='vis_bigeye',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for pose visualization and simple image '
                'effects. Have fun!', '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"s": Sunglasses effect B-)', '"b": Big-eye effect 0_0',
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
            out_video_file='webcam_demo.mp4',
            input_buffer='display',
            output_buffer='_display_'
            # `_display_` is an executor-reserved buffer
        )
    ])
