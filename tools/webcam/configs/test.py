# Copyright (c) OpenMMLab. All rights reserved.
runner = dict(
    name='Debug CamRunner',
    camera_id=0,
    display_delay=0,
    frame_buffer_size=20,
    user_buffers=[('result', 1)],
    nodes=[
        dict(
            type='DetectorNode',
            name='detector',
        ),
        dict(
            type='PoseVisualizerNode',
            name='visualizer',
            frame_buffer='_frame_',
            output_buffer='_output_')
    ])
