# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    # Basic configurations of the runner
    name='Super Saiyan Effects',
    camera_id=0,
    camera_fps=30,
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
        # 'DetectorNode':
        # This node performs object detection from the frame image using an
        # MMDetection model.
        dict(
            type='DetectorNode',
            name='Detector',
            model_config='demo/mmdetection_cfg/mask_rcnn_r50_fpn_2x_coco.py',
            model_checkpoint='https://download.openmmlab.com/'
            'mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/'
            'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392'
            '__segm_mAP-0.354_20200505_003907-3e542a40.pth',
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
            model_checkpoint='https://openmmlab-share.oss-cn-hangz'
            'hou.aliyuncs.com/mmpose/top_down/vipnas/vipnas_mbv3_co'
            'co_wholebody_256x192_dark-e2158108_20211205.pth',
            cls_names=['person'],
            input_buffer='det_result',
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
        # 'SaiyanNode':
        # This node draw the Super Saiyan effect in the frame image.
        # Pose results is needed.
        dict(
            type='SaiyanNode',
            name='Visualizer',
            enable_key='s',
            cls_names=['person'],
            enable=True,
            frame_buffer='frame',
            output_buffer='vis_saiyan'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='Helper',
            enable_key='h',
            frame_buffer='vis_saiyan',
            output_buffer='vis',
            content_lines=[
                'This is a demo for super saiyan effects. Have fun!', '',
                'Hot-keys:', '"s": Saiyan effect',
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
            frame_buffer='vis',
            output_buffer='_display_')  # `_frame_` is a runner-reserved buffer
    ])
