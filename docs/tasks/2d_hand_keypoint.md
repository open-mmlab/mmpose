# 2D Hand Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [OneHand10K](#onehand10k) \[ [Homepage](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html) \]
- [FreiHand](#freihand-dataset) \[ [Homepage](https://lmb.informatik.uni-freiburg.de/projects/freihand/) \]
- [CMU Panoptic HandDB](#cmu-panoptic-handdb) \[ [Homepage](http://domedb.perception.cs.cmu.edu/handdb.html) \]
- [InterHand2.6M](#interhand26m) \[ [Homepage](https://mks0601.github.io/InterHand2.6M/) \]
- [RHD](#rhd-dataset) \[ [Homepage](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) \]

## OneHand10K

<!-- [DATASET] -->

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

```bibtex
@inproceedings{zimmermann2019freihand,
  title={Freihand: A dataset for markerless capture of hand pose and shape from single rgb images},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={813--822},
  year={2019}
}
```

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
    │── onehand10k
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

```bibtex
@inproceedings{simon2017hand,
  title={Hand keypoint detection in single images using multiview bootstrapping},
  author={Simon, Tomas and Joo, Hanbyul and Matthews, Iain and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={1145--1153},
  year={2017}
}
```

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

```bibtex
@article{moon2020interhand2,
  title={InterHand2.6M: A dataset and baseline for 3D interacting hand pose estimation from a single RGB image},
  author={Moon, Gyeongsik and Yu, Shoou-I and Wen, He and Shiratori, Takaaki and Lee, Kyoung Mu},
  journal={arXiv preprint arXiv:2008.09309},
  year={2020},
  publisher={Springer}
}
```

For [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/), please download from [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/).
Please download the annotation files from [annotations](https://github.com/facebookresearch/InterHand2.6M/releases/download/v0.0/InterHand2.6M.annotations.5.fps.zip).
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

```bibtex
@TechReport{zb2017hand,
  author    = {Christian Zimmermann and Thomas Brox},
  title     = {Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution    = {arXiv:1705.01389},
  year      = {2017},
  note         = "https://arxiv.org/abs/1705.01389",
  url          = "https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

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
