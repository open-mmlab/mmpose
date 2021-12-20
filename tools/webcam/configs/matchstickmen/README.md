# MatchStickMen and ElkHorn Effects

This demo performs matchstickmen and elkhorn effects on video with pose estimation results, which highlight human keypoints and skeletons, and add heart and elkhorn in the frame image.

<div align="center">
    <img src="https://user-images.githubusercontent.com/87690686/149058108-b370e6ee-a48d-4132-bf26-57f222586405.gif" width="600px" alt><br>
</div>

## Instruction

### Get started

Launch the demo from the mmpose root directory:

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/matchstickmen/matchstickmen.py
```

### What's More

Try gesturing a heart with your hands and see what will happen ?:eyes:

### Hotkeys

| Hotkey | Function |
| -- | -- |
| n | Toggle the MatchStickMen effect on/off. |
| c | Toggle the ElkHorn effect on/off. |
| h | Toggle the instruction on/off. |
| m | Show the monitoring information. |
| q | Exit. |n

### Configuration

See the [README](/tools/webcam/configs/examples/README.md#configuration) of pose estimation demo for model configurations.

#### Image Preparation

- You can download a red heart image and elkhorn image then rename and place it like this: `'demo/resources/heart.png'`, `demo/resources/elk_horn.jpg` . You can also use the url link to the image.

- Then, modify the `src_img_path` field in the config file `matchstickmen.py` like this:

    ```python
    # 'MatchStickMenNode':
    # This node highlights human keypoints and skeletons in the image.
    # It can also launch a dynamically expanding red heart from the middle of
    # hands if the person is posing a "hand heart" gesture.
    # Pose results is needed.
    dict(
        type='MatchStickMenNode',
        name='Visualizer',
        enable_key='n',
        src_img_path = 'https://user-images.githubusercontent.com/87690686/149731850-ea946766-a4e8-4efa-82f5-e2f0515db8ae.png',
        frame_buffer='frame',
        output_buffer='vis_matchstickmen',
        background_color=(0, 0, 0),
    ),
    # 'ELkHornNode':
    # This node add an elkhorn on top of the person in the image.
    # Pose results is needed.
    dict(
        type='ELkHornNode',
        name='Visualizer',
        enable_key='c',
        src_img_path='https://user-images.githubusercontent.com/87690686/149731877-1a7ff0f3-fc5a-4fd5-b330-7f35e2930f02.jpg',
        frame_buffer='vis_matchstickmen',
        output_buffer='vis_elk_horn'),
    ```
