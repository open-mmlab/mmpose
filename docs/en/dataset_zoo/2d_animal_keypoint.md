# 2D Animal Keypoint Dataset

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [Animal-Pose](#animal-pose) \[ [Homepage](https://sites.google.com/view/animal-pose/) \]
- [AP-10K](#ap-10k) \[ [Homepage](https://github.com/AlexTheBad/AP-10K/) \]
- [Horse-10](#horse-10) \[ [Homepage](http://www.mackenziemathislab.org/horse10) \]
- [MacaquePose](#macaquepose) \[ [Homepage](http://pri.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html) \]
- [Vinegar Fly](#vinegar-fly) \[ [Homepage](https://github.com/jgraving/DeepPoseKit-Data) \]
- [Desert Locust](#desert-locust) \[ [Homepage](https://github.com/jgraving/DeepPoseKit-Data) \]
- [Grévy’s Zebra](#grvys-zebra) \[ [Homepage](https://github.com/jgraving/DeepPoseKit-Data) \]
- [ATRW](#atrw) \[ [Homepage](https://cvwc2019.github.io/challenge.html) \]
- [Animal Kingdom](#Animal-Kindom) \[ [Homepage](https://openaccess.thecvf.com/content/CVPR2022/html/Ng_Animal_Kingdom_A_Large_and_Diverse_Dataset_for_Animal_Behavior_CVPR_2022_paper.html) \]

## Animal-Pose

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html">Animal-Pose (ICCV'2019)</a></summary>

```bibtex
@InProceedings{Cao_2019_ICCV,
    author = {Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing},
    title = {Cross-Domain Adaptation for Animal Pose Estimation},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227796953-95ae1e30-5323-43f8-9a19-c4c2326e9835.png" height="200px">
</div>

For [Animal-Pose](https://sites.google.com/view/animal-pose/) dataset, we prepare the dataset as follows:

1. Download the images of [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data), especially the five categories (dog, cat, sheep, cow, horse), which we use as trainval dataset.
2. Download the [test-set](https://drive.google.com/drive/folders/1DwhQobZlGntOXxdm7vQsE4bqbFmN3b9y?usp=sharing) images with raw annotations (1000 images, 5 categories).
3. We have pre-processed the annotations to make it compatible with MMPose. Please download the annotation files from [annotations](https://download.openmmlab.com/mmpose/datasets/animalpose_annotations.tar). If you would like to generate the annotations by yourself, please check our dataset parsing [codes](/tools/dataset_converters/parse_animalpose_dataset.py).

Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── animalpose
        │
        │-- VOC2012
        │   │-- Annotations
        │   │-- ImageSets
        │   │-- JPEGImages
        │   │-- SegmentationClass
        │   │-- SegmentationObject
        │
        │-- animalpose_image_part2
        │   │-- cat
        │   │-- cow
        │   │-- dog
        │   │-- horse
        │   │-- sheep
        │
        │-- annotations
        │   │-- animalpose_train.json
        │   |-- animalpose_val.json
        │   |-- animalpose_trainval.json
        │   │-- animalpose_test.json
        │
        │-- PASCAL2011_animal_annotation
        │   │-- cat
        │   │   |-- 2007_000528_1.xml
        │   │   |-- 2007_000549_1.xml
        │   │   │-- ...
        │   │-- cow
        │   │-- dog
        │   │-- horse
        │   │-- sheep
        │
        │-- annimalpose_anno2
        │   │-- cat
        │   │   |-- ca1.xml
        │   │   |-- ca2.xml
        │   │   │-- ...
        │   │-- cow
        │   │-- dog
        │   │-- horse
        │   │-- sheep
```

The official dataset does not provide the official train/val/test set split.
We choose the images from PascalVOC for train & val. In total, we have 3608 images and 5117 annotations for train+val, where
2798 images with 4000 annotations are used for training, and 810 images with 1117 annotations are used for validation.
Those images from other sources (1000 images with 1000 annotations) are used for testing.

## AP-10K

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2108.12617">AP-10K (NeurIPS'2021)</a></summary>

```bibtex
@misc{yu2021ap10k,
      title={AP-10K: A Benchmark for Animal Pose Estimation in the Wild},
      author={Hang Yu and Yufei Xu and Jing Zhang and Wei Zhao and Ziyu Guan and Dacheng Tao},
      year={2021},
      eprint={2108.12617},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227797151-091dc21a-d944-49c9-8b62-cc47fa89e69f.png" height="200px">
</div>

For [AP-10K](https://github.com/AlexTheBad/AP-10K/) dataset, images and annotations can be downloaded from [download](https://drive.google.com/file/d/1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad/view?usp=sharing).
Note, this data and annotation data is for non-commercial use only.

Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── ap10k
        │-- annotations
        │   │-- ap10k-train-split1.json
        │   |-- ap10k-train-split2.json
        │   |-- ap10k-train-split3.json
        │   │-- ap10k-val-split1.json
        │   |-- ap10k-val-split2.json
        │   |-- ap10k-val-split3.json
        │   |-- ap10k-test-split1.json
        │   |-- ap10k-test-split2.json
        │   |-- ap10k-test-split3.json
        │-- data
        │   │-- 000000000001.jpg
        │   │-- 000000000002.jpg
        │   │-- ...
```

The annotation files in 'annotation' folder contains 50 labeled animal species. There are total 10,015 labeled images with 13,028 instances in the AP-10K dataset. We randonly split them into train, val, and test set following the ratio of 7:1:2.

## Horse-10

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html">Horse-10 (WACV'2021)</a></summary>

```bibtex
@inproceedings{mathis2021pretraining,
  title={Pretraining boosts out-of-domain robustness for pose estimation},
  author={Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1859--1868},
  year={2021}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227797934-32bc1b2c-7957-4a29-94df-8e431842ab3b.png" height="200px">
</div>

For [Horse-10](http://www.mackenziemathislab.org/horse10) dataset, images can be downloaded from [download](http://www.mackenziemathislab.org/horse10).
Please download the annotation files from [horse10_annotations](https://download.openmmlab.com/mmpose/datasets/horse10_annotations.tar). Note, this data and annotation data is for non-commercial use only, per the authors (see http://horse10.deeplabcut.org for more information).
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

<details>
<summary align="right"><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/pmc7874091/">MacaquePose (bioRxiv'2020)</a></summary>

```bibtex
@article{labuguen2020macaquepose,
  title={MacaquePose: A novel ‘in the wild’macaque monkey pose dataset for markerless motion capture},
  author={Labuguen, Rollyn and Matsumoto, Jumpei and Negrete, Salvador and Nishimaru, Hiroshi and Nishijo, Hisao and Takada, Masahiko and Go, Yasuhiro and Inoue, Ken-ichi and Shibata, Tomohiro},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227799576-f10f8469-9432-4139-beb4-195037dee72c.png" height="200px">
</div>

For [MacaquePose](http://pri.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html) dataset, images can be downloaded from [download](http://pri.ehub.kyoto-u.ac.jp/datasets/macaquepose/download.php).
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

Since the official dataset does not provide the test set, we randomly select 12500 images for training, and the rest for evaluation (see [code](/tools/dataset/parse_macaquepose_dataset.py)).

## Vinegar Fly

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.nature.com/articles/s41592-018-0234-5">Vinegar Fly (Nature Methods'2019)</a></summary>

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

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227802774-bb4e4ef2-2ade-42ad-80f1-97f2a7faa9e2.png" height="200px">
</div>

For [Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data) dataset, images can be downloaded from [vinegar_fly_images](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_images.tar).
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

Since the official dataset does not provide the test set, we randomly select 90% images for training, and the rest (10%) for evaluation (see [code](/tools/dataset_converters/parse_deepposekit_dataset.py)).

## Desert Locust

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://elifesciences.org/articles/47994">Desert Locust (Elife'2019)</a></summary>

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

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227802779-09d0ec8c-8971-4c67-a315-e2d1355f7f72.png" height="200px">
</div>

For [Desert Locust](https://github.com/jgraving/DeepPoseKit-Data) dataset, images can be downloaded from [locust_images](https://download.openmmlab.com/mmpose/datasets/locust_images.tar).
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
    │── locust
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

Since the official dataset does not provide the test set, we randomly select 90% images for training, and the rest (10%) for evaluation (see [code](/tools/dataset_converters/parse_deepposekit_dataset.py)).

## Grévy’s Zebra

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://elifesciences.org/articles/47994">Grévy’s Zebra (Elife'2019)</a></summary>

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

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227802783-ace952bb-1ff9-4720-80a8-c63cc9e714b6.png" height="200px">
</div>
For [Grévy’s Zebra](https://github.com/jgraving/DeepPoseKit-Data) dataset, images can be downloaded from [zebra_images](https://download.openmmlab.com/mmpose/datasets/zebra_images.tar).
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

Since the official dataset does not provide the test set, we randomly select 90% images for training, and the rest (10%) for evaluation (see [code](/tools/dataset_converters/parse_deepposekit_dataset.py)).

## ATRW

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1906.05586">ATRW (ACM MM'2020)</a></summary>

```bibtex
@inproceedings{li2020atrw,
  title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
  author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2590--2598},
  year={2020}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227797386-fce99241-8a0e-4a40-a179-dad013e6c5a4.png" height="200px">
</div>
ATRW captures images of the Amur tiger (also known as Siberian tiger, Northeast-China tiger) in the wild.
For [ATRW](https://cvwc2019.github.io/challenge.html) dataset, please download images from
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

## Animal Kingdom

<details>
<summary align="right"><a href="https://arxiv.org/abs/2204.08129">Animal Kingdom (CVPR'2022)</a></summary>
</details>
<div align="center">
  <img src="https://github.com/open-mmlab/mmpose/assets/53283758/8591989e-91fa-4f6e-99c8-48b614de862e" height="200px">
</div>

```bibtex
@inproceedings{Ng_2022_CVPR,
    author    = {Ng, Xun Long and Ong, Kian Eng and Zheng, Qichen and Ni, Yun and Yeo, Si Yong and Liu, Jun},
    title     = {Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19023-19034}
 }
```

For [Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom) dataset, images can be downloaded from [here](https://forms.office.com/pages/responsepage.aspx?id=drd2NJDpck-5UGJImDFiPVRYpnTEMixKqPJ1FxwK6VZUQkNTSkRISTNORUI2TDBWMUpZTlQ5WUlaSyQlQCN0PWcu).
Please Extract dataset under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── ak
        |--annotations
        │  │-- ak_P1
        │  │   │-- train.json
        │  │   │-- test.json
        │  │-- ak_P2
        │  │   │-- train.json
        │  │   │-- test.json
        │  │-- ak_P3_amphibian
        │  │   │-- train.json
        │  │   │-- test.json
        │  │-- ak_P3_bird
        │  │   │-- train.json
        │  │   │-- test.json
        │  │-- ak_P3_fish
        │  │   │-- train.json
        │  │   │-- test.json
        │  │-- ak_P3_mammal
        │  │   │-- train.json
        │  │   │-- test.json
        │  │-- ak_P3_reptile
        │      │-- train.json
        │      │-- test.json
        │-- images
        │   │-- AAACXZTV
        │   │   │--AAACXZTV_f000059.jpg
        │   │   │--...
        │   │-- AAAUILHH
        │   │   │--AAAUILHH_f000098.jpg
        │   │   │--...
        │   │-- ...
```
