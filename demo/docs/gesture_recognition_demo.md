## Hand Gesture Recognition Demo

We provide a demo for gesture recognition with MMPose. This demo is built upon [MMPose Webcam API](/docs/en/tutorials/7_webcam_api.md).

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/172213082-afb9d71a-f2df-4509-932c-e47dc61ec7d7.gif" width="600px" alt><br>
</div>

### Get started

Launch the demo from the mmpose root directory:

```shell
python demo/webcam_demo.py --config demo/webcam_cfg/gesture_recognition.py
```

### Hotkeys

| Hotkey | Function                                                    |
| ------ | ----------------------------------------------------------- |
| v      | Toggle the gesture recognition result visualization on/off. |
| h      | Show help information.                                      |
| m      | Show the monitoring information.                            |
| q      | Exit.                                                       |

Note that the demo will automatically save the output video into a file `gesture.mp4`.

### Configurations

Detailed configurations can be found in the [config file](/demo/webcam_cfg/gesture_recognition.py). And more information about the gesture recognition model used in the demo can be found at the [model page](/configs/hand/gesture_sview_rgbd_vid/mtut/nvgesture/i3d_nvgesture.md).
