# 2D Hand Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [OneHand10K](#onehand10k) \[ [Homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html) \]
- [FreiHand](#freihand-dataset) \[ [Homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/) \]
- [CMU Panoptic HandDB](#cmu-panoptic-handdb) \[ [Homepage](http://domedb.perception.cs.cmu.edu/handdb.html) \]
- [InterHand2.6M](#interhand26m) \[ [Homepage](https://mks0601.github.io/InterHand2.6M/) \]
- [RHD](#rhd-dataset) \[ [Homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) \]
- [COCO-WholeBody-Hand](#coco-wholebody-hand) \[ [Homepage](https://github.com/jin-s13/COCO-WholeBody/) \]

## OneHand10K

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

For [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html) data, please download from [OneHand10K Dataset](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html).
Please download the annotation files from [onehand10k_annotations](https://download.openmmlab.com/mmpose/datasets/onehand10k_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── onehand10k
        |── annotations
        |   |── onehand10k_train.json
        |   |── onehand10k_test.json
        `── Train
        |   |── source
        |       |── 0.jpg
        |       |── 1.jpg
        |        ...
        `── Test
            |── source
                |── 0.jpg
                |── 1.jpg

```

## FreiHAND Dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Zimmermann_FreiHAND_A_Dataset_for_Markerless_Capture_of_Hand_Pose_and_ICCV_2019_paper.html">FreiHand (ICCV'2019)</a></summary>

```bibtex
@inproceedings{zimmermann2019freihand,
  title={Freihand: A dataset for markerless capture of hand pose and shape from single rgb images},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={813--822},
  year={2019}
}
```

</details>

For [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/) data, please download from [FreiHand Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
Since the official dataset does not provide validation set, we randomly split the training data into 8:1:1 for train/val/test.
Please download the annotation files from [freihand_annotations](https://download.openmmlab.com/mmpose/datasets/frei_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── freihand
        |── annotations
        |   |── freihand_train.json
        |   |── freihand_val.json
        |   |── freihand_test.json
        `── training
            |── rgb
            |   |── 00000000.jpg
            |   |── 00000001.jpg
            |    ...
            |── mask
                |── 00000000.jpg
                |── 00000001.jpg
                 ...
```

## CMU Panoptic HandDB

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.html">CMU Panoptic HandDB (CVPR'2017)</a></summary>

```bibtex
@inproceedings{simon2017hand,
  title={Hand keypoint detection in single images using multiview bootstrapping},
  author={Simon, Tomas and Joo, Hanbyul and Matthews, Iain and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={1145--1153},
  year={2017}
}
```

</details>

For [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html), please download from [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html).
Following [Simon et al](https://arxiv.org/abs/1704.07809), panoptic images (hand143_panopticdb) and MPII & NZSL training sets (manual_train) are used for training, while MPII & NZSL test set (manual_test) for testing.
Please download the annotation files from [panoptic_annotations](https://download.openmmlab.com/mmpose/datasets/panoptic_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── panoptic
        |── annotations
        |   |── panoptic_train.json
        |   |── panoptic_test.json
        |
        `── hand143_panopticdb
        |   |── imgs
        |   |   |── 00000000.jpg
        |   |   |── 00000001.jpg
        |   |    ...
        |
        `── hand_labels
            |── manual_train
            |   |── 000015774_01_l.jpg
            |   |── 000015774_01_r.jpg
            |    ...
            |
            `── manual_test
                |── 000648952_02_l.jpg
                |── 000835470_01_l.jpg
                 ...
```

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
Please download the annotation files from [annotations](https://drive.google.com/drive/folders/1pWXhdfaka-J0fSAze0MsajN0VpZ8e8tO).
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

## RHD Dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://lmb.informatik.uni-freiburg.de/projects/hand3d/">RHD (ICCV'2017)</a></summary>

```bibtex
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

</details>

For [RHD Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html), please download from [RHD Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html).
Please download the annotation files from [rhd_annotations](https://download.openmmlab.com/mmpose/datasets/rhd_annotations.zip).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── rhd
        |── annotations
        |   |── rhd_train.json
        |   |── rhd_test.json
        `── training
        |   |── color
        |   |   |── 00000.jpg
        |   |   |── 00001.jpg
        |   |── depth
        |   |   |── 00000.jpg
        |   |   |── 00001.jpg
        |   |── mask
        |   |   |── 00000.jpg
        |   |   |── 00001.jpg
        `── evaluation
        |   |── color
        |   |   |── 00000.jpg
        |   |   |── 00001.jpg
        |   |── depth
        |   |   |── 00000.jpg
        |   |   |── 00001.jpg
        |   |── mask
        |   |   |── 00000.jpg
        |   |   |── 00001.jpg
```

## COCO-WholeBody (Hand)

[DATASET]

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

For [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/) dataset, images can be downloaded from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download COCO-WholeBody annotations for COCO-WholeBody annotations for [Train](https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf/view?usp=sharing) / [Validation](https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb/view?usp=sharing) (Google Drive).
Download person detection result of COCO val2017 from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- annotations
        │   │-- coco_wholebody_train_v1.0.json
        │   |-- coco_wholebody_val_v1.0.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...
```

Please also install the latest version of [Extended COCO API](https://github.com/jin-s13/xtcocoapi) to support COCO-WholeBody evaluation:

`pip install xtcocotools`
