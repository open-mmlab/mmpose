# Meow Dwen Dwen

Do you know [Bing DwenDwen (冰墩墩)](https://en.wikipedia.org/wiki/Bing_Dwen_Dwen_and_Shuey_Rhon_Rhon), the mascot of 2022 Beijing Olympic Games?

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/152742707-c0c51844-e1d0-42d0-9a12-e369002e082f.jpg" width="224px" alt><br>
</div>

Now you can dress your cat up in this costume and TA-DA! Be prepared for super cute **Meow Dwen Dwen**.

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/152942243-a17194a2-0fd1-4467-993c-634f6d7966d8.gif" width="300px" alt><br>
</div>

You are a dog fan? Hold on, here comes Woof Dwen Dwen.

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/152942747-84240692-3944-48a5-b60b-e60bd0a4339c.gif" width="300px" alt><br>
</div>

## Instruction

### Get started

Launch the demo from the mmpose root directory:

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/meow_dwen_dwen/meow_dwen_dwen.py
```

### Hotkeys

| Hotkey | Function |
| -- | -- |
| s | Change the background. |
| h | Show help information. |
| m | Show diagnostic information. |
| q | Exit. |

### Configuration

- **Use video input**

As you can see in the config, we set `camera_id` as the path of the input image. You can also set it as a video file path (or url), or a webcam ID number (e.g. `camera_id=0`), to capture the dynamic face from the video input.
