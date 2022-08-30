# Useful Tools

Apart from training/testing scripts, We provide lots of useful tools under the `tools/` directory.

## Analysis Tools

<!-- TOC -->

- [Log Analysis](#log-analysis)
- [Model Complexity (experimental)](#model-complexity-experimental)
- [Benchmarks (experimental)](#benchmarks)
- [Print the entire config](#print-the-entire-config)

<!-- TOC -->

### Log Analysis

`tools/analysis/analyze_logs.py` plots loss/pose acc curves given a training log file. Run `pip install seaborn` first to install the dependency.

![acc_curve_image](https://user-images.githubusercontent.com/26127467/187380669-d126299e-1d9c-43cf-9acf-6f3ee138bfce.png)

```shell
python tools/analysis/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

- Plot the mse loss of some run.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys loss --legend loss
  ```

- Plot the acc of some run, and save the figure to a pdf.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log.json --keys acc_pose --out results.pdf
  ```

- Compare the acc of two runs in the same figure.

  ```shell
  python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys acc_pose --legend run1 run2
  ```

You can also compute the average training speed.

```shell
python tools/analysis/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
```

- Compute the average training speed for a config file

  ```shell
  python tools/analysis/analyze_logs.py cal_train_time log.json
  ```

  The output is expected to be like the following.

  ```text
  -----Analyze train time of log.json-----
  slowest epoch 114, average time is 0.9662
  fastest epoch 16, average time is 0.7532
  time std over epochs is 0.0426
  average iter time: 0.8406 s/iter
  ```

### Model Complexity (Experimental)

`/tools/analysis/get_flops.py` is a script adapted from [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) to compute the FLOPs and params of a given model.

```shell
python tools/analysis/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

We will get the result like this

```
==============================
Input shape: (1, 3, 256, 192)
Flops: 8.9 GMac
Params: 28.04 M
==============================
```

```{note}
This tool is still experimental and we do not guarantee that the number is absolutely correct.
```

You may use the result for simple comparisons, but double check it before you adopt it in technical reports or papers.

(1) FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 340, 256) for 2D recognizer, (1, 3, 32, 340, 256) for 3D recognizer.
(2) Some operators are not counted into FLOPs like GN and custom operators. Refer to [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py) for details.

### Benchmarks

#### Inference

`tools/analysis/benchmark_inference.py` tests inference speed of the model specified by config.

```shell
python tools/analysis/benchmark_inference.py ${CONFIG} [-h] [--fuse-conv-bn] [--log-interval ${LOG_INTERVAL}]
```

Users can fuse the convolutional layers and succeeding BatchNorm layers using the option `--fuse-conv-bn`. This will slightly increase the inference speed.

#### Data Processing

`tools/analysis/benchmark_processing.py` tests speed of the data processing.

```shell
python tools/analysis/benchmark_processing.py ${CONFIG} [-h]
```

### Print the entire config

`tools/analysis/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/analysis/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

## Dataset Tools

<!-- TOC -->

- [Animal Pose](#animal-pose)
- [COFW](#cofw)
- [DeepposeKit (Fly, Locust, Zebra)](#deepposekit)
- [Macaque](#macaque)
- [H36M](#human36m)
- [MPII](#mpii)
- [MPI-INF-3DHP](#mpi-inf-3dhp)

<!-- TOC -->

### Animal Pose

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

For [Animal-Pose](https://sites.google.com/view/animal-pose/) dataset, the images and annotations can be downloaded from [official website](https://sites.google.com/view/animal-pose/). The script `tools/dataset_converters/parse_animalpose_dataset.py` converts raw annotations into the format compatible with MMPose. The pre-processed [annotation files](https://download.openmmlab.com/mmpose/datasets/animalpose_annotations.tar) are available. If you would like to generate the annotations by yourself, please follows:

1. Download the raw images and annotations and extract them under `{MMPose}/data`. Make them look like this:

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

   The generated annotation files are put in `{MMPose}/data/animalpose/annotations`.

The official dataset does not provide the official train/val/test set split.
We choose the images from PascalVOC for train & val. In total, we have 3608 images and 5117 annotations for train+val, where
2798 images with 4000 annotations are used for training, and 810 images with 1117 annotations are used for validation.
Those images from other sources (1000 images with 1000 annotations) are used for testing.

### COFW

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
Move `COFW_train_color.mat` and `COFW_test_color.mat` to `data/cofw/` and make them look like:

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

Run the following script under `{MMPose}/data`

`python tools/dataset_converters/parse_cofw_dataset.py`

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

### DeepposeKit

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

1. Download the raw images and annotations and extract them under `{MMPose}/data`. Make them look like this:

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

   The generated annotation files are put in `{MMPose}/data/fly/annotations`, `{MMPose}/data/locust/annotations`, and `{MMPose}/data/zebra/annotations`.

Since the official dataset does not provide the test set, we randomly select 90% images for training, and the rest (10%) for evaluation.

### Macaque

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

For [MacaquePose](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) dataset, images and annotations can be downloaded from [download](http://www2.ehub.kyoto-u.ac.jp/datasets/macaquepose/index.html). The script `tools/dataset_converters/parse_macaquepose_dataset.py` converts raw annotations into the format compatible with MMPose. The pre-processed [macaque_annotations](https://download.openmmlab.com/mmpose/datasets/macaque_annotations.tar) are available. If you would like to generate the annotations by yourself, please follows:

1. Download the raw images and annotations and extract them under `{MMPose}/data`. Make them look like this:

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

   The generated annotation files are put in `{MMPose}/data/macaque/annotations`.

Since the official dataset does not provide the test set, we randomly select 12500 images for training, and the rest for evaluation.

### Human36M

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

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php), please download from the official website and place the files under `{MMPose}/data/h36m`.
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
python /tools/dataset_converters/h36m_to_coco.py
```

### MPII

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

During training and inference for [MPII dataset](<[MPII](http://human-pose.mpi-inf.mpg.de/)>), the prediction result will be saved as '.mat' format by default. We also provide a tool to convert this '.mat' to more readable '.json' format.

```shell
python tools/dataset_converters/mat2json ${PRED_MAT_FILE} ${GT_JSON_FILE} ${OUTPUT_PRED_JSON_FILE}
```

For example,

```shell
python tools/dataset/mat2json work_dirs/res50_mpii_256x256/pred.mat data/mpii/annotations/mpii_val.json pred.json
```

### MPII-INF-3DHP

Work in progress...
