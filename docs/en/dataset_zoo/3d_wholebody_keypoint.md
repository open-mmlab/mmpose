# 3D Body Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [H3WB](#h3wb) \[ [Homepage](https://github.com/wholebody3d/wholebody3d) \]

## H3WB

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2211.15692">H3WB (ICCV'2023)</a></summary>

```bibtex
@InProceedings{Zhu_2023_ICCV,
    author    = {Zhu, Yue and Samet, Nermin and Picard, David},
    title     = {H3WB: Human3.6M 3D WholeBody Dataset and Benchmark},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20166-20177}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227770977-c8f00355-c43a-467e-8444-d307789cf4b2.png" height="300px">
</div>

For [H3WB](https://github.com/wholebody3d/wholebody3d), please follow the [document](3d_body_keypoint.md#human36m) to download [Human3.6M](http://vision.imar.ro/human3.6m/description.php) dataset, then download the H3WB annotations from the official [webpage](https://github.com/wholebody3d/wholebody3d). The data should have the following structure:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── h36m
        ├── annotation_body3d
        |   ├── cameras.pkl
        |   ├── fps50
        |   |   ├── h36m_test.npz
        |   |   ├── h36m_train.npz
        |   |   ├── joint2d_rel_stats.pkl
        |   |   ├── joint2d_stats.pkl
        |   |   ├── joint3d_rel_stats.pkl
        |   |   `── joint3d_stats.pkl
        |   `── fps10
        |       ├── h36m_test.npz
        |       ├── h36m_train.npz
        |       ├── joint2d_rel_stats.pkl
        |       ├── joint2d_stats.pkl
        |       ├── joint3d_rel_stats.pkl
        |       `── joint3d_stats.pkl
        `── images
            ├── S1
            |   ├── S1_Directions_1.54138969
            |   |   ├── S1_Directions_1.54138969_00001.jpg
            |   |   ├── S1_Directions_1.54138969_00002.jpg
            |   |   ├── ...
            |   ├── ...
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            `── S11
        `── json
            ├── 2Dto3D_test_2d.json
            ├── 2Dto3D_train.json
            ├── 2Dto3D_train_part1.json
            ├── 2Dto3D_train_part2.json
            ├── 2Dto3D_train_part3.json
            ├── 2Dto3D_train_part4.json
            ├── 2Dto3D_train_part5.json
            ├── I2Dto3D_test_2d.json
            ├── RGBto3D_test_img.json
            └── RGBto3D_train.json
```
