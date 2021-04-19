# 2D Fashion Landmark Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [DeepFashion](#deepfashion) \[ [Homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html) \]

## DeepFashion (Fashion Landmark Detection, FLD)

<!-- [DATASET] -->

```bibtex
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{liuYLWTeccv16FashionLandmark,
 author = {Liu, Ziwei and Yan, Sijie and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 title = {Fashion Landmark Detection in the Wild},
 booktitle = {European Conference on Computer Vision (ECCV)},
 month = {October},
 year = {2016}
 }
```

For [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html) datatset, images can be downloaded from [download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html).
Please download the annotation files from [fld_annotations](https://download.openmmlab.com/mmpose/datasets/fld_annotations.tar).
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
        │   │-- fld_upper_train.json
        │   |-- fld_upper_val.json
        │   |-- fld_upper_test.json
        │   │-- fld_lower_train.json
        │   |-- fld_lower_val.json
        │   |-- fld_lower_test.json
        │   │-- fld_full_train.json
        │   |-- fld_full_val.json
        │   |-- fld_full_test.json
        │-- img
        │   │-- img_00000001.jpg
        │   │-- img_00000002.jpg
        │   │-- img_00000003.jpg
        │   │-- img_00000004.jpg
        │   │-- img_00000005.jpg
        │   │-- ...
```
