# 2D Face Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [300W](#300w-dataset) \[ [Homepage](https://ibug.doc.ic.ac.uk/resources/300-W/) \]
- [WFLW](#wflw-dataset) \[ [Homepage](https://wywu.github.io/projects/LAB/WFLW.html) \]
- [AFLW](#aflw-dataset) \[ [Homepage](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) \]
- [COFW](#cofw-dataset) \[ [Homepage](http://www.vision.caltech.edu/xpburgos/ICCV13/) \]
- [COCO-WholeBody-Face](#coco-wholebody-face) \[ [Homepage](https://github.com/jin-s13/COCO-WholeBody/) \]
- [LaPa](#lapa-dataset) \[ [Homepage](https://github.com/JDAI-CV/lapa-dataset) \]

## 300W Dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.sciencedirect.com/science/article/pii/S0262885616000147">300W (IMAVIS'2016)</a></summary>

```bibtex
@article{sagonas2016300,
  title={300 faces in-the-wild challenge: Database and results},
  author={Sagonas, Christos and Antonakos, Epameinondas and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  journal={Image and vision computing},
  volume={47},
  pages={3--18},
  year={2016},
  publisher={Elsevier}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227780043-e557acc3-5966-48ba-ac9e-b6c5916f1e55.jpg" height="200px">
</div>

For 300W data, please download images from [300W Dataset](https://ibug.doc.ic.ac.uk/resources/300-W/).
Please download the annotation files from [300w_annotations](https://download.openmmlab.com/mmpose/datasets/300w_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── 300w
        |── annotations
        |   |── face_landmarks_300w_train.json
        |   |── face_landmarks_300w_valid.json
        |   |── face_landmarks_300w_valid_common.json
        |   |── face_landmarks_300w_valid_challenge.json
        |   |── face_landmarks_300w_test.json
        `── images
            |── afw
            |   |── 1051618982_1.jpg
            |   |── 111076519_1.jpg
            |    ...
            |── helen
            |   |── trainset
            |   |   |── 100032540_1.jpg
            |   |   |── 100040721_1.jpg
            |   |    ...
            |   |── testset
            |   |   |── 296814969_3.jpg
            |   |   |── 2968560214_1.jpg
            |   |    ...
            |── ibug
            |   |── image_003_1.jpg
            |   |── image_004_1.jpg
            |    ...
            |── lfpw
            |   |── trainset
            |   |   |── image_0001.png
            |   |   |── image_0002.png
            |   |    ...
            |   |── testset
            |   |   |── image_0001.png
            |   |   |── image_0002.png
            |   |    ...
            `── Test
                |── 01_Indoor
                |   |── indoor_001.png
                |   |── indoor_002.png
                |    ...
                `── 02_Outdoor
                    |── outdoor_001.png
                    |── outdoor_002.png
                     ...
```

## WFLW Dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Look_at_Boundary_CVPR_2018_paper.html">WFLW (CVPR'2018)</a></summary>

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227785100-4b3d1e39-0d64-47c0-9e55-c947ce70866e.png" height="200px">
</div>

For WFLW data, please download images from [WFLW Dataset](https://wywu.github.io/projects/LAB/WFLW.html).
Please download the annotation files from [wflw_annotations](https://download.openmmlab.com/mmpose/datasets/wflw_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── wflw
        |── annotations
        |   |── face_landmarks_wflw_train.json
        |   |── face_landmarks_wflw_test.json
        |   |── face_landmarks_wflw_test_blur.json
        |   |── face_landmarks_wflw_test_occlusion.json
        |   |── face_landmarks_wflw_test_expression.json
        |   |── face_landmarks_wflw_test_largepose.json
        |   |── face_landmarks_wflw_test_illumination.json
        |   |── face_landmarks_wflw_test_makeup.json
        |
        `── images
            |── 0--Parade
            |   |── 0_Parade_marchingband_1_1015.jpg
            |   |── 0_Parade_marchingband_1_1031.jpg
            |    ...
            |── 1--Handshaking
            |   |── 1_Handshaking_Handshaking_1_105.jpg
            |   |── 1_Handshaking_Handshaking_1_107.jpg
            |    ...
            ...
```

## AFLW Dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6130513/">AFLW (ICCVW'2011)</a></summary>

```bibtex
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```

</details>

For AFLW data, please download images from [AFLW Dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/).
Please download the annotation files from [aflw_annotations](https://download.openmmlab.com/mmpose/datasets/aflw_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── aflw
        |── annotations
        |   |── face_landmarks_aflw_train.json
        |   |── face_landmarks_aflw_test_frontal.json
        |   |── face_landmarks_aflw_test.json
        `── images
            |── flickr
                |── 0
                |   |── image00002.jpg
                |   |── image00013.jpg
                |    ...
                |── 2
                |   |── image00004.jpg
                |   |── image00006.jpg
                |    ...
                `── 3
                    |── image00032.jpg
                    |── image00035.jpg
                     ...
```

## COFW Dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_iccv_2013/html/Burgos-Artizzu_Robust_Face_Landmark_2013_ICCV_paper.html">COFW (ICCV'2013)</a></summary>

```bibtex
@inproceedings{burgos2013robust,
  title={Robust face landmark estimation under occlusion},
  author={Burgos-Artizzu, Xavier P and Perona, Pietro and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1513--1520},
  year={2013}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227786792-06604943-c062-4bcd-bb2d-a2f78d80115b.png" height="200px">
</div>

For COFW data, please download from [COFW Dataset (Color Images)](http://www.vision.caltech.edu/xpburgos/ICCV13/Data/COFW_color.zip).
Move `COFW_train_color.mat` and `COFW_test_color.mat` to `data/cofw/` and make them look like:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── cofw
        |── COFW_train_color.mat
        |── COFW_test_color.mat
```

Run the following script under `{MMPose}/data`

`python tools/dataset_converters/parse_cofw_dataset.py`

And you will get

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── cofw
        |── COFW_train_color.mat
        |── COFW_test_color.mat
        |── annotations
        |   |── cofw_train.json
        |   |── cofw_test.json
        |── images
            |── 000001.jpg
            |── 000002.jpg
```

## COCO-WholeBody (Face)

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227787217-2f226dc0-e5d7-4d0b-9ab8-68b53a5467c2.png" height="200px">
</div>

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

## LaPa

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://aaai.org/ojs/index.php/AAAI/article/view/6832/6686">LaPa (AAAI'2020)</a></summary>

```bibtex
@inproceedings{liu2020new,
  title={A New Dataset and Boundary-Attention Semantic Segmentation for Face Parsing.},
  author={Liu, Yinglu and Shi, Hailin and Shen, Hao and Si, Yue and Wang, Xiaobo and Mei, Tao},
  booktitle={AAAI},
  pages={11637--11644},
  year={2020}
}
```

</details>

<div align="center">
  <img src="https://github.com/lucia123/lapa-dataset/raw/master/sample.png" height="200px">
</div>

For [LaPa](https://github.com/JDAI-CV/lapa-dataset) dataset, images can be downloaded from [their github page](https://github.com/JDAI-CV/lapa-dataset).

Download and extract them under $MMPOSE/data, and use our `tools/dataset_converters/lapa2coco.py` to make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── LaPa
        │-- annotations
        │   │-- lapa_train.json
        │   |-- lapa_val.json
        │   |-- lapa_test.json
        |   |-- lapa_trainval.json
        │-- train
        │   │-- images
        │   │-- labels
        │   │-- landmarks
        │-- val
        │   │-- images
        │   │-- labels
        │   │-- landmarks
        `-- test
        │   │-- images
        │   │-- labels
        │   │-- landmarks

```
