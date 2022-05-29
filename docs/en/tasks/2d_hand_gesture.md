# 2D Hand Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [NVGesture](#nvgesture) \[ [Homepage](https://www.v7labs.com/open-datasets/nvgesture) \]

## NVGesture

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d">OneHand10K (CVPR'2016)</a></summary>

```bibtex
@inproceedings{molchanov2016online,
  title={Online detection and classification of dynamic hand gestures with recurrent 3d convolutional neural network},
  author={Molchanov, Pavlo and Yang, Xiaodong and Gupta, Shalini and Kim, Kihwan and Tyree, Stephen and Kautz, Jan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4207--4215},
  year={2016}
}
```

</details>

For [NVGesture](https://www.v7labs.com/open-datasets/nvgesture) data and annotation, please download from [NVGesture Dataset](https://drive.google.com/drive/folders/0ByhYoRYACz9cMUk0QkRRMHM3enc?resourcekey=0-cJe9M3PZy2qCbfGmgpFrHQ&usp=sharing).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── nvgesture
        |── annotations
        |   |── nvgesture_train_correct_cvpr2016_v2.lst
        |   |── nvgesture_test_correct_cvpr2016_v2.lst
        |       ...
        `── Video_data
            |── class_01
            |   |── subject1_r0
            |   |   |── sk_color.avi
            |   |   |── sk_depth.avi
            |   |       ...
            |   |── subject1_r1
            |   |── subject2_r0
            |       ...
            |── class_02
            |── class_03
                ...

```

The hand bounding box is computed by the hand detection model described in [det model zoo](/demo/docs/mmdet_modelzoo.md). The detected bounding box can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1AGOeX0iHhaigBVRicjetieNRC7Zctuz4?usp=sharing). It is recommended to place it at `data/nvgesture/annotations/bboxes.json`.
