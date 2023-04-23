# 2D Fashion Landmark Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [DeepFashion2](#deepfashion2) \[ [Homepage](https://github.com/switchablenorms/DeepFashion2) \]

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
