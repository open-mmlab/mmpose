# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    name='Debug CamRunner',
    camera_id=0,
    camera_fps=30,
    user_buffers=[('display', 30)],
    nodes=[
        dict(
            type='MonitorNode',
            name='Monitor',
            enable_key='m',
            style='fancy',
            frame_buffer='_frame_',
            output_buffer='_display_'),
        # dict(
        #     type='RecorderNode',
        #     name='Recorder',
        #     out_video_file='webcam_output.mp4',
        #     frame_buffer='display',
        #     output_buffer='_display_')
    ])
