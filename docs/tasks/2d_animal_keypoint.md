# 2D Animal Keypoint Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [Horse-10](#horse-10) \[ [Homepage](http://www.mackenziemathislab.org/horse10) \]
- [MacaquePose](#macaquepose) \[ [Homepage](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) \]

## Horse-10

[DATASET]

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

[DATASET]

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
