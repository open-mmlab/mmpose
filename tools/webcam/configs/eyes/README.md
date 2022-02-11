# Sunglasses and Bug-eye Effects

We can apply fun effects on videos with pose estimation results, like adding sunglasses on the face, or make the eyes look bigger.

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/124059525-ce20c580-da5d-11eb-8e4a-2d96cd31fe9f.gif" width="600px" alt><br>
</div>

## Instruction

### Get started

Launch the demo from the mmpose root directory:

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/examples/pose_estimation.py
```

### Hotkeys

| Hotkey | Function |
| -- | -- |
| s | Toggle the sunglasses effect on/off. |
| b | Toggle the bug-eye effect on/off. |
| h | Show help information. |
| m | Show the monitoring information. |
| q | Exit. |

### Configuration

See the [README](/tools/webcam/configs/examples/README.md#configuration) of pose estimation demo for model configurations.
