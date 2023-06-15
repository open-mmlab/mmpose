# 数据集格式转换脚本

MMPose 提供了一些工具来帮助用户处理数据集。

## Animal Pose 数据集

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

对于 [Animal-Pose](https://sites.google.com/view/animal-pose/)，可以从[官方网站](https://sites.google.com/view/animal-pose/)下载图像和标注。脚本 `tools/dataset_converters/parse_animalpose_dataset.py` 将原始标注转换为 MMPose 兼容的格式。预处理的[标注文件](https://download.openmmlab.com/mmpose/datasets/animalpose_annotations.tar)可用。如果您想自己生成标注，请按照以下步骤操作：

1. 下载图片与标注信息并解压到 `$MMPOSE/data`，按照以下格式组织：

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

2. 运行脚本

   ```bash
   python tools/dataset_converters/parse_animalpose_dataset.py
   ```

   生成的标注文件将保存在 `$MMPOSE/data/animalpose/annotations` 中。

开源作者没有提供官方的 train/val/test 划分，我们选择来自 PascalVOC 的图片作为 train & val，train+val 一共 3600 张图片，5117 个标注。其中 2798 张图片，4000 个标注用于训练，810 张图片，1117 个标注用于验证。测试集包含 1000 张图片，1000 个标注用于评估。

## COFW 数据集

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

对于 COFW 数据集，请从 [COFW Dataset (Color Images)](https://data.caltech.edu/records/20099) 进行下载。

将 `COFW_train_color.mat` 和 `COFW_test_color.mat` 移动到 `$MMPOSE/data/cofw/`，确保它们按照以下格式组织：

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

运行 `pip install h5py` 安装依赖，然后在 `$MMPOSE` 下运行脚本：

```bash
python tools/dataset_converters/parse_cofw_dataset.py
```

最终结果为：

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

## DeepposeKit 数据集

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

对于 [Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data)，[Desert Locust](https://github.com/jgraving/DeepPoseKit-Data), 和 [Grévy’s Zebra](https://github.com/jgraving/DeepPoseKit-Data) 数据集，请从 [DeepPoseKit-Data](https://github.com/jgraving/DeepPoseKit-Data) 下载数据。

`tools/dataset_converters/parse_deepposekit_dataset.py` 脚本可以将原始标注转换为 MMPose 支持的格式。我们已经转换好的标注文件可以在这里下载：

- [vinegar_fly_annotations](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_annotations.tar)
- [locust_annotations](https://download.openmmlab.com/mmpose/datasets/locust_annotations.tar)
- [zebra_annotations](https://download.openmmlab.com/mmpose/datasets/zebra_annotations.tar)

如果你希望自己转换数据，请按照以下步骤操作：

1. 下载原始图片和标注，并解压到 `$MMPOSE/data`，将它们按照以下格式组织：

   ```text
   mmpose
   ├── mmpose
   ├── docs
   ├── tests
   ├── tools
   ├── configs
   `── data
       |
       |── DeepPoseKit-Data
       |   `── datasets
       |       |── fly
       |       |   |── annotation_data_release.h5
       |       |   |── skeleton.csv
       |       |   |── ...
       |       |
       |       |── locust
       |       |   |── annotation_data_release.h5
       |       |   |── skeleton.csv
       |       |   |── ...
       |       |
       |       `── zebra
       |           |── annotation_data_release.h5
       |           |── skeleton.csv
       |           |── ...
       |
       │── fly
           `-- images
               │-- 0.jpg
               │-- 1.jpg
               │-- ...
   ```

   图片也可以在 [vinegar_fly_images](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_images.tar)，[locust_images](https://download.openmmlab.com/mmpose/datasets/locust_images.tar) 和[zebra_images](https://download.openmmlab.com/mmpose/datasets/zebra_images.tar) 下载。

2. 运行脚本：

   ```bash
   python tools/dataset_converters/parse_deepposekit_dataset.py
   ```

   生成的标注文件将保存在 $MMPOSE/data/fly/annotations`，`$MMPOSE/data/locust/annotations`和`$MMPOSE/data/zebra/annotations\` 中。

由于官方数据集中没有提供测试集，我们随机选择了 90% 的图片用于训练，剩下的 10% 用于测试。

## Macaque 数据集

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

对于 [MacaquePose](http://www2.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html) 数据集，请从 [这里](http://www2.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html) 下载数据。

`tools/dataset_converters/parse_macaquepose_dataset.py` 脚本可以将原始标注转换为 MMPose 支持的格式。我们已经转换好的标注文件可以在 [这里](https://download.openmmlab.com/mmpose/datasets/macaque_annotations.tar) 下载。

如果你希望自己转换数据，请按照以下步骤操作：

1. 下载原始图片和标注，并解压到 `$MMPOSE/data`，将它们按照以下格式组织：

   ```text
   mmpose
   ├── mmpose
   ├── docs
   ├── tests
   ├── tools
   ├── configs
   `── data
       │── macaque
           │-- annotations.csv
           │-- images
           │   │-- 01418849d54b3005.jpg
           │   │-- 0142d1d1a6904a70.jpg
           │   │-- 01ef2c4c260321b7.jpg
           │   │-- 020a1c75c8c85238.jpg
           │   │-- 020b1506eef2557d.jpg
           │   │-- ...
   ```

2. 运行脚本：

   ```bash
   python tools/dataset_converters/parse_macaquepose_dataset.py
   ```

   生成的标注文件将保存在 `$MMPOSE/data/macaque/annotations` 中。

由于官方数据集中没有提供测试集，我们随机选择了 90% 的图片用于训练，剩下的 10% 用于测试。

## Human3.6M 数据集

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>

对于 [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 数据集，请从官网下载数据，放置到 `$MMPOSE/data/h36m` 下。

然后执行 [预处理脚本](/tools/dataset_converters/preprocess_h36m.py)。

```bash
python tools/dataset_converters/preprocess_h36m.py --metadata {path to metadata.xml} --original data/h36m
```

这将在全帧率（50 FPS）和降频帧率（10 FPS）下提取相机参数和姿势注释。处理后的数据应具有以下结构：

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── h36m
        ├── annotation_body3d
        |   ├── cameras.pkl
        |   ├── fps50
        |   |   ├── h36m_test.npz
        |   |   ├── h36m_train.npz
        |   |   ├── joint2d_rel_stats.pkl
        |   |   ├── joint2d_stats.pkl
        |   |   ├── joint3d_rel_stats.pkl
        |   |   `── joint3d_stats.pkl
        |   `── fps10
        |       ├── h36m_test.npz
        |       ├── h36m_train.npz
        |       ├── joint2d_rel_stats.pkl
        |       ├── joint2d_stats.pkl
        |       ├── joint3d_rel_stats.pkl
        |       `── joint3d_stats.pkl
        `── images
            ├── S1
            |   ├── S1_Directions_1.54138969
            |   |   ├── S1_Directions_1.54138969_00001.jpg
            |   |   ├── S1_Directions_1.54138969_00002.jpg
            |   |   ├── ...
            |   ├── ...
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            `── S11
```

然后，标注信息需要转换为 MMPose 支持的 COCO 格式。这可以通过运行以下命令完成：

```bash
python tools/dataset_converters/h36m_to_coco.py
```

## MPII 数据集

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

对于 [MPII](http://human-pose.mpi-inf.mpg.de/) 数据集，请从官网下载数据，放置到 `$MMPOSE/data/mpii` 下。

我们提供了一个脚本来将 `.mat` 格式的标注文件转换为 `.json` 格式。这可以通过运行以下命令完成：

```shell
python tools/dataset_converters/mat2json ${PRED_MAT_FILE} ${GT_JSON_FILE} ${OUTPUT_PRED_JSON_FILE}
```

例如：

```shell
python tools/dataset/mat2json work_dirs/res50_mpii_256x256/pred.mat data/mpii/annotations/mpii_val.json pred.json
```

## Label Studio 数据集

<details>
<summary align="right"><a href="https://github.com/heartexlabs/label-studio/">Label Studio</a></summary>

```bibtex
@misc{Label Studio,
  title={{Label Studio}: Data labeling software},
  url={https://github.com/heartexlabs/label-studio},
  note={Open source software available from https://github.com/heartexlabs/label-studio},
  author={
    Maxim Tkachenko and
    Mikhail Malyuk and
    Andrey Holmanyuk and
    Nikolai Liubimov},
  year={2020-2022},
}
```

</details>

对于 [Label Studio](https://github.com/heartexlabs/label-studio/) 用户，请依照 [Label Studio 转换工具文档](./label_studio.md) 中的方法进行标注，并将结果导出为 Label Studio 标准的 `.json` 文件，将 `Labeling Interface` 中的 `Code` 保存为 `.xml` 文件。

我们提供了一个脚本来将 Label Studio 标准的 `.json` 格式标注文件转换为 COCO 标准的 `.json` 格式。这可以通过运行以下命令完成：

```shell
python tools/dataset_converters/labelstudio2coco.py ${LS_JSON_FILE} ${LS_XML_FILE} ${OUTPUT_COCO_JSON_FILE}
```

例如：

```shell
python tools/dataset_converters/labelstudio2coco.py config.xml project-1-at-2023-05-13-09-22-91b53efa.json output/result.json
```
