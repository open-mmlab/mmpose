## Webcam Demo

We provide a webcam demo tool which integrartes detection and 2D pose estimation for humans and animals. You can simply run the following command:

```python
python demo/webcam_demo.py
```

It will launch a window to display the webcam video steam with detection and pose estimation results:

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/124059525-ce20c580-da5d-11eb-8e4a-2d96cd31fe9f.gif" width="600px" alt><br>
</div>

### Usage Tips

- **Which model is used in the demo tool?**

  Please check the following default arguments in the script. You can also choose other models from the [MMDetection Model Zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md) and [MMPose Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html#) or use your own models.

  | Model | Arguments |
  | :--: | :--  |
  | Detection | `--det-config`, `--det-checkpoint` |
  | Human Pose | `--human-pose-config`, `--human-pose-checkpoint` |
  | Animal Pose | `--animal-pose-config`, `--animal-pose-checkpoint` |

- **Can this tool run without GPU?**

  Yes, you can set `--device=cpu` and the model inference will be performed on CPU. Of course, this may cause a low inference FPS compared to using GPU devices.

- **Why there is time delay between the pose visualization and the video?**

  The video I/O and model inference are running asynchronously and the latter usually takes more time for a single frame. To allevidate the time delay, you can:

  1. set `--display-delay=MILLISECONDS` to defer the video stream, according to the inference delay shown at the top left corner. Or,

  2. set `--synchronous-mode` to force video stream being aligned with inference results. This may reduce the video display FPS.

- **Can this tool process video files?**

  Yes. You can set `--cam-id=VIDEO_FILE_PATH` to run the demo tool in offline mode on a video file. Note that `--synchronous-mode` should be set in this case.

- **How to enable/disable the special effects?**

  The special effects can be enabled/disabled at launch time by setting arguments like `--bugeye`, `--sunglasses`, *etc*. You can also toggle the effects by keyboard shortcuts like `b`, `s` when the tool starts.

- **What if my computer doesn't have a camera?**

  You can use a smart phone as a webcam with apps like [Camo](https://reincubate.com/camo/) or [DroidCam](https://www.dev47apps.com/).
