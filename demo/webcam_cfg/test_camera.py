# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    name='Test Webcam',
    camera_id=0,
    camera_max_fps=30,
    nodes=[
        dict(
            type='MonitorNode',
            name='monitor',
            enable_key='m',
            enable=False,
            input_buffer='_frame_',
            output_buffer='display'),
        # 'RecorderNode':
        # This node save the output video into a file.
        dict(
            type='RecorderNode',
            name='recorder',
            out_video_file='webcam_output.mp4',
            input_buffer='display',
            output_buffer='_display_')
    ])
