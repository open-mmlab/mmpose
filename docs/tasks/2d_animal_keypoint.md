# 2D Animal Keypoint Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [Horse-10](#horse-10) \[ [Homepage](http://www.mackenziemathislab.org/horse10) \]
- [MacaquePose](#macaquepose) \[ [Homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) \]
- [Vinegar Fly](#vinegar-fly) \[ [Homepage](https://github.com/jgraving/DeepPoseKit-Data) \]
- [Desert Locust](#desert-locust) \[ [Homepage](https://github.com/jgraving/DeepPoseKit-Data) \]
- [Grévy’s Zebra](#grvys-zebra) \[ [Homepage](https://github.com/jgraving/DeepPoseKit-Data) \]
- [ATRW](#atrw) \[ [Homepage](https://cvwc2019.github.io/challenge.html) \]

## Horse-10

<!-- [DATASET] -->

```bibtex
@inproceedings{mathis2021pretraining,
  title={Pretraining boosts out-of-domain robustness for pose estimation},
  author={Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1859--1868},
  year={2021}
}
```

For [Horse-10](http://www.mackenziemathislab.org/horse10) datatset, images can be downloaded from [download](http://www.mackenziemathislab.org/horse10).
Please download the annotation files from [horse10_annotations](https://download.openmmlab.com/mmpose/datasets/horse10_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── horse10
        │-- annotations
        │   │-- horse10-train-split1.json
        │   |-- horse10-train-split2.json
        │   |-- horse10-train-split3.json
        │   │-- horse10-test-split1.json
        │   |-- horse10-test-split2.json
        │   |-- horse10-test-split3.json
        │-- labeled-data
        │   │-- BrownHorseinShadow
        │   │-- BrownHorseintoshadow
        │   │-- ...

```

## MacaquePose

<!-- [DATASET] -->

```bibtex
@article{labuguen2020macaquepose,
  title={MacaquePose: A novel ‘in the wild’macaque monkey pose dataset for markerless motion capture},
  author={Labuguen, Rollyn and Matsumoto, Jumpei and Negrete, Salvador and Nishimaru, Hiroshi and Nishijo, Hisao and Takada, Masahiko and Go, Yasuhiro and Inoue, Ken-ichi and Shibata, Tomohiro},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

For [MacaquePose](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) datatset, images can be downloaded from [download](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html).
Please download the annotation files from [macaque_annotations](https://download.openmmlab.com/mmpose/datasets/macaque_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── macaque
        │-- annotations
        │   │-- macaque_train.json
        │   |-- macaque_test.json
        │-- images
        │   │-- 01418849d54b3005.jpg
        │   │-- 0142d1d1a6904a70.jpg
        │   │-- 01ef2c4c260321b7.jpg
        │   │-- 020a1c75c8c85238.jpg
        │   │-- 020b1506eef2557d.jpg
        │   │-- ...

```

Since the official dataset does not provide the test set, we randomly select 12500 images for training, and the rest for evaluation (see [code](tools/dataset/parse_macaquepose_dataset.py)).

## Vinegar Fly

<!-- [DATASET] -->

```bibtex
@article{pereira2019fast,
  title={Fast animal pose estimation using deep neural networks},
  author={Pereira, Talmo D and Aldarondo, Diego E and Willmore, Lindsay and Kislin, Mikhail and Wang, Samuel S-H and Murthy, Mala and Shaevitz, Joshua W},
  journal={Nature methods},
  volume={16},
  number={1},
  pages={117--125},
  year={2019},
  publisher={Nature Publishing Group}
}
```

For [Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data) datatset, images can be downloaded from [vinegar_fly_images](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_images.tar).
Please download the annotation files from [vinegar_fly_annotations](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── fly
        │-- annotations
        │   │-- fly_train.json
        │   |-- fly_test.json
        │-- images
        │   │-- 0.jpg
        │   │-- 1.jpg
        │   │-- 2.jpg
        │   │-- 3.jpg
        │   │-- ...

```

Since the official dataset does not provide the test set, we randomly select 90\% images for training, and the rest (10\%) for evaluation (see [code](tools/dataset/parse_deepposekit_dataset.py)).

## Desert Locust

<!-- [DATASET] -->

```bibtex
@article{graving2019deepposekit,
  title={DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning},
  author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
  journal={Elife},
  volume={8},
  pages={e47994},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```

For [Desert Locust](https://github.com/jgraving/DeepPoseKit-Data) datatset, images can be downloaded from [locust_images](https://download.openmmlab.com/mmpose/datasets/locust_images.tar).
Please download the annotation files from [locust_annotations](https://download.openmmlab.com/mmpose/datasets/locust_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── fly
        │-- annotations
        │   │-- locust_train.json
        │   |-- locust_test.json
        │-- images
        │   │-- 0.jpg
        │   │-- 1.jpg
        │   │-- 2.jpg
        │   │-- 3.jpg
        │   │-- ...

```

Since the official dataset does not provide the test set, we randomly select 90\% images for training, and the rest (10\%) for evaluation (see [code](tools/dataset/parse_deepposekit_dataset.py)).

## Grévy’s Zebra

<!-- [DATASET] -->

```bibtex
@article{graving2019deepposekit,
  title={DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning},
  author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
  journal={Elife},
  volume={8},
  pages={e47994},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```

For [Grévy’s Zebra](https://github.com/jgraving/DeepPoseKit-Data) datatset, images can be downloaded from [zebra_images](https://download.openmmlab.com/mmpose/datasets/zebra_images.tar).
Please download the annotation files from [zebra_annotations](https://download.openmmlab.com/mmpose/datasets/zebra_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── zebra
        │-- annotations
        │   │-- zebra_train.json
        │   |-- zebra_test.json
        │-- images
        │   │-- 0.jpg
        │   │-- 1.jpg
        │   │-- 2.jpg
        │   │-- 3.jpg
        │   │-- ...

```

Since the official dataset does not provide the test set, we randomly select 90\% images for training, and the rest (10\%) for evaluation (see [code](tools/dataset/parse_deepposekit_dataset.py)).

## ATRW

<!-- [DATASET] -->

```bibtex
@inproceedings{li2020atrw,
  title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
  author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2590--2598},
  year={2020}
}
```

ATRW captures images of the Amur tiger (also known as Siberian tiger, Northeast-China tiger) in the wild.
For [ATRW](https://cvwc2019.github.io/challenge.html) datatset, please download images from
[Pose_train](https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_pose_train.tar.gz),
[Pose_val](https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_pose_val.tar.gz), and
[Pose_test](https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_pose_test.tar.gz).
Note that in the ATRW official annotation files, the key "file_name" is written as "filename". To make it compatible with
other coco-type json files, we have modified this key.
Please download the modified annotation files from [atrw_annotations](https://download.openmmlab.com/mmpose/datasets/atrw_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── atrw
        │-- annotations
        │   │-- keypoint_train.json
        │   │-- keypoint_val.json
        │   │-- keypoint_trainval.json
        │-- images
        │   │-- train
        │   │   │-- 000002.jpg
        │   │   │-- 000003.jpg
        │   │   │-- ...
        │   │-- val
        │   │   │-- 000001.jpg
        │   │   │-- 000013.jpg
        │   │   │-- ...
        │   │-- test
        │   │   │-- 000000.jpg
        │   │   │-- 000004.jpg
        │   │   │-- ...

```
