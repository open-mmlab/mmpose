# 3D Hand Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [InterHand2.6M](#interhand26m) \[ [Homepage](https://mks0601.github.io/InterHand2.6M/) \]

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
