# Dataset Tools

## Animal Pose

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

For [Animal-Pose](https://sites.google.com/view/animal-pose/) dataset, the images and annotations can be downloaded from [official website](https://sites.google.com/view/animal-pose/). The script `tools/dataset_converters/parse_animalpose_dataset.py` converts raw annotations into the format compatible with MMPose. The pre-processed [annotation files](https://download.openmmlab.com/mmpose/datasets/animalpose_annotations.tar) are available. If you would like to generate the annotations by yourself, please follow:

1. Download the raw images and annotations and extract them under `$MMPOSE/data`. Make them look like this:

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

2. Run command

   ```bash
   python tools/dataset_converters/parse_animalpose_dataset.py
   ```

   The generated annotation files are put in `$MMPOSE/data/animalpose/annotations`.

The official dataset does not provide the official train/val/test set split.
We choose the images from PascalVOC for train & val. In total, we have 3608 images and 5117 annotations for train+val, where
2798 images with 4000 annotations are used for training, and 810 images with 1117 annotations are used for validation.
Those images from other sources (1000 images with 1000 annotations) are used for testing.

## COFW

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

For COFW data, please download from [COFW Dataset (Color Images)](https://data.caltech.edu/records/20099).
Move `COFW_train_color.mat` and `COFW_test_color.mat` to `$MMPOSE/data/cofw/` and make them look like:

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

Run `pip install h5py` first to install the dependency, then run the following script under `$MMPOSE`:

```bash
python tools/dataset_converters/parse_cofw_dataset.py
```

And you will get

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

## DeepposeKit

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

For [Vinegar Fly](https://github.com/jgraving/DeepPoseKit-Data), [Desert Locust](https://github.com/jgraving/DeepPoseKit-Data), and [Grévy’s Zebra](https://github.com/jgraving/DeepPoseKit-Data) dataset, the annotations files can be downloaded from [DeepPoseKit-Data](https://github.com/jgraving/DeepPoseKit-Data). The script `tools/dataset_converters/parse_deepposekit_dataset.py` converts raw annotations into the format compatible with MMPose. The pre-processed annotation files are available at [vinegar_fly_annotations](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_annotations.tar), [locust_annotations](https://download.openmmlab.com/mmpose/datasets/locust_annotations.tar), and [zebra_annotations](https://download.openmmlab.com/mmpose/datasets/zebra_annotations.tar). If you would like to generate the annotations by yourself, please follows:

1. Download the raw images and annotations and extract them under `$MMPOSE/data`. Make them look like this:

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

   Note that the images can be downloaded from [vinegar_fly_images](https://download.openmmlab.com/mmpose/datasets/vinegar_fly_images.tar), [locust_images](https://download.openmmlab.com/mmpose/datasets/locust_images.tar), and [zebra_images](https://download.openmmlab.com/mmpose/datasets/zebra_images.tar).

2. Run command

   ```bash
   python tools/dataset_converters/parse_deepposekit_dataset.py
   ```

   The generated annotation files are put in `$MMPOSE/data/fly/annotations`, `$MMPOSE/data/locust/annotations`, and `$MMPOSE/data/zebra/annotations`.

Since the official dataset does not provide the test set, we randomly select 90% images for training, and the rest (10%) for evaluation.

## Macaque

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

For [MacaquePose](http://www2.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html) dataset, images and annotations can be downloaded from [download](http://www2.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html). The script `tools/dataset_converters/parse_macaquepose_dataset.py` converts raw annotations into the format compatible with MMPose. The pre-processed [macaque_annotations](https://download.openmmlab.com/mmpose/datasets/macaque_annotations.tar) are available. If you would like to generate the annotations by yourself, please follows:

1. Download the raw images and annotations and extract them under `$MMPOSE/data`. Make them look like this:

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

2. Run command

   ```bash
   python tools/dataset_converters/parse_macaquepose_dataset.py
   ```

   The generated annotation files are put in `$MMPOSE/data/macaque/annotations`.

Since the official dataset does not provide the test set, we randomly select 12500 images for training, and the rest for evaluation.

## Human3.6M

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

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php), please download from the official website and place the files under `$MMPOSE/data/h36m`.
Then run the [preprocessing script](/tools/dataset_converters/preprocess_h36m.py):

```bash
python tools/dataset_converters/preprocess_h36m.py --metadata {path to metadata.xml} --original data/h36m
```

This will extract camera parameters and pose annotations at full framerate (50 FPS) and downsampled framerate (10 FPS). The processed data should have the following structure:

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

After that, the annotations need to be transformed into COCO format which is compatible with MMPose. Please run:

```bash
python tools/dataset_converters/h36m_to_coco.py
```

## MPII

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

During training and inference for [MPII](http://human-pose.mpi-inf.mpg.de/), the prediction result will be saved as '.mat' format by default. We also provide a tool to convert this `.mat` to more readable `.json` format.

```shell
python tools/dataset_converters/mat2json ${PRED_MAT_FILE} ${GT_JSON_FILE} ${OUTPUT_PRED_JSON_FILE}
```

For example,

```shell
python tools/dataset/mat2json work_dirs/res50_mpii_256x256/pred.mat data/mpii/annotations/mpii_val.json pred.json
```

## Label Studio

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

For users of [Label Studio](https://github.com/heartexlabs/label-studio/), please follow the instructions in the [Label Studio to COCO document](./label_studio.md) to annotate and export the results as a Label Studio `.json` file. And save the `Code` from the `Labeling Interface` as an `.xml` file.

We provide a script to convert Label Studio `.json` annotation file to COCO `.json` format file. It can be used by running the following command:

```shell
python tools/dataset_converters/labelstudio2coco.py ${LS_JSON_FILE} ${LS_XML_FILE} ${OUTPUT_COCO_JSON_FILE}
```

For example,

```shell
python tools/dataset_converters/labelstudio2coco.py config.xml project-1-at-2023-05-13-09-22-91b53efa.json output/result.json
```
