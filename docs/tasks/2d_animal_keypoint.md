# 2D Animal Keypoint Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [Horse-10](#horse-10) \[ [Homepage](http://www.mackenziemathislab.org/horse10) \]

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
    │── fld
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
