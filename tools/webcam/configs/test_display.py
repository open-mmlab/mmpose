# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    name='Debug CamRunner',
    camera_id=0,
    camera_fps=30,
    ms_display_delay=0,
    user_buffer=[],
    nodes=[
        dict(
            type='MonitorNode',
            name='Monitor',
            enable_key='m',
            style='fancy',
            input_buffer='_frame_',
            output_buffer='_display_')
    ])
