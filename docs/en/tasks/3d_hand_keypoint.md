# 3D Hand Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [InterHand2.6M](#interhand26m) \[ [Homepage](https://mks0601.github.io/InterHand2.6M/) \]

## InterHand2.6M

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/content/pdf/10.1007/978-3-030-58565-5_33.pdf">InterHand2.6M (ECCV'2020)</a></summary>

```bibtex
@InProceedings{Moon_2020_ECCV_InterHand2.6M,
author = {Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
title = {InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2020}
}
```

</details>

For [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/), please download from [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/).
Please download the annotation files from [annotations](https://download.openmmlab.com/mmpose/datasets/interhand2.6m_annotations.zip).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── interhand2.6m
        |── annotations
        |   |── all
        |   |── human_annot
        |   |── machine_annot
        |   |── skeleton.txt
        |   |── subject.txt
        |
        `── images
        |   |── train
        |   |   |-- Capture0 ~ Capture26
        |   |── val
        |   |   |-- Capture0
        |   |── test
        |   |   |-- Capture0 ~ Capture7
```
