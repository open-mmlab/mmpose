# 2D Fashion Landmark Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [DeepFashion](#deepfashion) \[ [Homepage](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html) \]
- [DeepFashion2](#deepfashion2) \[ [Homepage](https://github.com/switchablenorms/DeepFashion2) \]

## DeepFashion (Fashion Landmark Detection, FLD)

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html">DeepFashion (CVPR'2016)</a></summary>

```bibtex
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-46475-6_15">DeepFashion (ECCV'2016)</a></summary>

```bibtex
@inproceedings{liuYLWTeccv16FashionLandmark,
 author = {Liu, Ziwei and Yan, Sijie and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 title = {Fashion Landmark Detection in the Wild},
 booktitle = {European Conference on Computer Vision (ECCV)},
 month = {October},
 year = {2016}
 }
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227774588-443fc5cc-7842-472a-abd5-827f0e3fd27f.png" height="150px">
</div>

For [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html) dataset, images can be downloaded from [download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html).
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

## DeepFashion2

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1901.07973.pdf">DeepFashion2 (CVPR'2019)</a></summary>

```bibtex
@article{DeepFashion2,
  author = {Yuying Ge and Ruimao Zhang and Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo},
  title={A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images},
  journal={CVPR},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

For [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) dataset, images can be downloaded from [download](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing).
Please download the [annotation files](https://drive.google.com/file/d/1RM9l9EaB9ULRXhoCS72PkCXtJ4Cn4i6O/view?usp=share_link). These annotation files are converted by [deepfashion2_to_coco.py](https://github.com/switchablenorms/DeepFashion2/blob/master/evaluation/deepfashion2_to_coco.py).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── deepfashion2
        │── train
            │-- deepfashion2_short_sleeved_outwear_train.json
            │-- deepfashion2_short_sleeved_dress_train.json
            │-- deepfashion2_skirt_train.json
            │-- deepfashion2_sling_dress_train.json
            │-- ...
            │-- image
            │   │-- 000001.jpg
            │   │-- 000002.jpg
            │   │-- 000003.jpg
            │   │-- 000004.jpg
            │   │-- 000005.jpg
            │   │-- ...
        │── validation
            │-- deepfashion2_short_sleeved_dress_validation.json
            │-- deepfashion2_long_sleeved_shirt_validation.json
            │-- deepfashion2_trousers_validation.json
            │-- deepfashion2_skirt_validation.json
            │-- ...
            │-- image
            │   │-- 000001.jpg
            │   │-- 000002.jpg
            │   │-- 000003.jpg
            │   │-- 000004.jpg
            │   │-- 000005.jpg
            │   │-- ...
```
