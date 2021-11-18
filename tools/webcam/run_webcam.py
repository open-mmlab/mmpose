# Copyright (c) OpenMMLab. All rights reserved.
from mmcv import Config
from webcam import WebcamRunner

if __name__ == '__main__':
    cfgs = dict(
        name='Camera Runner',
        camera_id=0,
        # camera_id='demo/resources/demo.mp4',
        display_delay=0,
        frame_buffer_size=30,
    )
    cfgs = Config(cfgs)
    runner = WebcamRunner(cfgs)

    runner.run()
