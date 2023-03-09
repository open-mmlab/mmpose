# 2D Body Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- Images
  - [COCO](#coco) \[ [Homepage](http://cocodataset.org/) \]
  - [MPII](#mpii) \[ [Homepage](http://human-pose.mpi-inf.mpg.de/) \]
  - [MPII-TRB](#mpii-trb) \[ [Homepage](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) \]
  - [AI Challenger](#aic) \[ [Homepage](https://github.com/AIChallenger/AI_Challenger_2017) \]
  - [CrowdPose](#crowdpose) \[ [Homepage](https://github.com/Jeff-sjtu/CrowdPose) \]
  - [OCHuman](#ochuman) \[ [Homepage](https://github.com/liruilong940607/OCHumanApi) \]
  - [MHP](#mhp) \[ [Homepage](https://lv-mhp.github.io/dataset) \]
- Videos
  - [PoseTrack18](#posetrack18) \[ [Homepage](https://posetrack.net/users/download.php) \]
  - [sub-JHMDB](#sub-jhmdb-dataset) \[ [Homepage](http://jhmdb.is.tue.mpg.de/dataset) \]

## COCO

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

For [COCO](http://cocodataset.org/) data, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results.
Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Optionally, to evaluate on COCO'2017 test-dev, please download the [image-info](https://download.openmmlab.com/mmpose/datasets/person_keypoints_test-dev-2017.json).
Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        │   |-- person_keypoints_test-dev-2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...

```

## MPII

<!-- [DATASET] -->

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

For [MPII](http://human-pose.mpi-inf.mpg.de/) data, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
We have converted the original annotation files into json format, please download them from [mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mpii
        |── annotations
        |   |── mpii_gt_val.mat
        |   |── mpii_test.json
        |   |── mpii_train.json
        |   |── mpii_trainval.json
        |   `── mpii_val.json
        `── images
            |── 000001163.jpg
            |── 000003072.jpg

```

During training and inference, the prediction result will be saved as '.mat' format by default. We also provide a tool to convert this '.mat' to more readable '.json' format.

```shell
python tools/dataset/mat2json ${PRED_MAT_FILE} ${GT_JSON_FILE} ${OUTPUT_PRED_JSON_FILE}
```

For example,

```shell
python tools/dataset/mat2json work_dirs/res50_mpii_256x256/pred.mat data/mpii/annotations/mpii_val.json pred.json
```

## MPII-TRB

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Duan_TRB_A_Novel_Triplet_Representation_for_Understanding_2D_Human_Body_ICCV_2019_paper.html">MPII-TRB (ICCV'2019)</a></summary>

```bibtex
@inproceedings{duan2019trb,
  title={TRB: A Novel Triplet Representation for Understanding 2D Human Body},
  author={Duan, Haodong and Lin, Kwan-Yee and Jin, Sheng and Liu, Wentao and Qian, Chen and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9479--9488},
  year={2019}
}
```

</details>

For [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) data, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
Please download the annotation files from [mpii_trb_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_trb_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mpii
        |── annotations
        |   |── mpii_trb_train.json
        |   |── mpii_trb_val.json
        `── images
            |── 000001163.jpg
            |── 000003072.jpg

```

## AIC

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1711.06475">AI Challenger (ArXiv'2017)</a></summary>

```bibtex
@article{wu2017ai,
  title={Ai challenger: A large-scale dataset for going deeper in image understanding},
  author={Wu, Jiahong and Zheng, He and Zhao, Bo and Li, Yixin and Yan, Baoming and Liang, Rui and Wang, Wenjia and Zhou, Shipei and Lin, Guosen and Fu, Yanwei and others},
  journal={arXiv preprint arXiv:1711.06475},
  year={2017}
}
```

</details>

For [AIC](https://github.com/AIChallenger/AI_Challenger_2017) data, please download from [AI Challenger 2017](https://github.com/AIChallenger/AI_Challenger_2017), 2017 Train/Val is needed for keypoints training and validation.
Please download the annotation files from [aic_annotations](https://download.openmmlab.com/mmpose/datasets/aic_annotations.tar).
Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── aic
        │-- annotations
        │   │-- aic_train.json
        │   |-- aic_val.json
        │-- ai_challenger_keypoint_train_20170902
        │   │-- keypoint_train_images_20170902
        │   │   │-- 0000252aea98840a550dac9a78c476ecb9f47ffa.jpg
        │   │   │-- 000050f770985ac9653198495ef9b5c82435d49c.jpg
        │   │   │-- ...
        `-- ai_challenger_keypoint_validation_20170911
            │-- keypoint_validation_images_20170911
                │-- 0002605c53fb92109a3f2de4fc3ce06425c3b61f.jpg
                │-- 0003b55a2c991223e6d8b4b820045bd49507bf6d.jpg
                │-- ...
```

## CrowdPose

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

For [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) data, please download from [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose).
Please download the annotation files and human detection results from [crowdpose_annotations](https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar).
For top-down approaches, we follow [CrowdPose](https://arxiv.org/abs/1812.00324) to use the [pre-trained weights](https://pjreddie.com/media/files/yolov3.weights) of [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) to generate the detected human bounding boxes.
For model training, we follow [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) to train models on CrowdPose train/val dataset, and evaluate models on CrowdPose test dataset.
Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── crowdpose
        │-- annotations
        │   │-- mmpose_crowdpose_train.json
        │   │-- mmpose_crowdpose_val.json
        │   │-- mmpose_crowdpose_trainval.json
        │   │-- mmpose_crowdpose_test.json
        │   │-- det_for_crowd_test_0.1_0.5.json
        │-- images
            │-- 100000.jpg
            │-- 100001.jpg
            │-- 100002.jpg
            │-- ...
```

## OCHuman

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Pose2Seg_Detection_Free_Human_Instance_Segmentation_CVPR_2019_paper.html">OCHuman (CVPR'2019)</a></summary>

```bibtex
@inproceedings{zhang2019pose2seg,
  title={Pose2seg: Detection free human instance segmentation},
  author={Zhang, Song-Hai and Li, Ruilong and Dong, Xin and Rosin, Paul and Cai, Zixi and Han, Xi and Yang, Dingcheng and Huang, Haozhi and Hu, Shi-Min},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={889--898},
  year={2019}
}
```

</details>

For [OCHuman](https://github.com/liruilong940607/OCHumanApi) data, please download the images and annotations from [OCHuman](https://github.com/liruilong940607/OCHumanApi),
Move them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── ochuman
        │-- annotations
        │   │-- ochuman_coco_format_val_range_0.00_1.00.json
        │   |-- ochuman_coco_format_test_range_0.00_1.00.json
        |-- images
            │-- 000001.jpg
            │-- 000002.jpg
            │-- 000003.jpg
            │-- ...

```

## MHP

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://dl.acm.org/doi/abs/10.1145/3240508.3240509">MHP (ACM MM'2018)</a></summary>

```bibtex
@inproceedings{zhao2018understanding,
  title={Understanding humans in crowded scenes: Deep nested adversarial learning and a new benchmark for multi-human parsing},
  author={Zhao, Jian and Li, Jianshu and Cheng, Yu and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={792--800},
  year={2018}
}
```

</details>

For [MHP](https://lv-mhp.github.io/dataset) data, please download from [MHP](https://lv-mhp.github.io/dataset).
Please download the annotation files from [mhp_annotations](https://download.openmmlab.com/mmpose/datasets/mhp_annotations.tar.gz).
Please download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mhp
        │-- annotations
        │   │-- mhp_train.json
        │   │-- mhp_val.json
        │
        `-- train
        │   │-- images
        │   │   │-- 1004.jpg
        │   │   │-- 10050.jpg
        │   │   │-- ...
        │
        `-- val
        │   │-- images
        │   │   │-- 10059.jpg
        │   │   │-- 10068.jpg
        │   │   │-- ...
        │
        `-- test
        │   │-- images
        │   │   │-- 1005.jpg
        │   │   │-- 10052.jpg
        │   │   │-- ...~~~~
```

## PoseTrack18

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Andriluka_PoseTrack_A_Benchmark_CVPR_2018_paper.html">PoseTrack18 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{andriluka2018posetrack,
  title={Posetrack: A benchmark for human pose estimation and tracking},
  author={Andriluka, Mykhaylo and Iqbal, Umar and Insafutdinov, Eldar and Pishchulin, Leonid and Milan, Anton and Gall, Juergen and Schiele, Bernt},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5167--5176},
  year={2018}
}
```

</details>

For [PoseTrack18](https://posetrack.net/users/download.php) data, please download from [PoseTrack18](https://posetrack.net/users/download.php).
Please download the annotation files from [posetrack18_annotations](https://download.openmmlab.com/mmpose/datasets/posetrack18_annotations.tar).
We have merged the video-wise separated official annotation files into two json files (posetrack18_train & posetrack18_val.json). We also generate the [mask files](https://download.openmmlab.com/mmpose/datasets/posetrack18_mask.tar) to speed up training.
For top-down approaches, we use [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) to generate the detected human bounding boxes.
Please download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── posetrack18
        │-- annotations
        │   │-- posetrack18_train.json
        │   │-- posetrack18_val.json
        │   │-- posetrack18_val_human_detections.json
        │   │-- train
        │   │   │-- 000001_bonn_train.json
        │   │   │-- 000002_bonn_train.json
        │   │   │-- ...
        │   │-- val
        │   │   │-- 000342_mpii_test.json
        │   │   │-- 000522_mpii_test.json
        │   │   │-- ...
        │   `-- test
        │       │-- 000001_mpiinew_test.json
        │       │-- 000002_mpiinew_test.json
        │       │-- ...
        │
        `-- images
        │   │-- train
        │   │   │-- 000001_bonn_train
        │   │   │   │-- 000000.jpg
        │   │   │   │-- 000001.jpg
        │   │   │   │-- ...
        │   │   │-- ...
        │   │-- val
        │   │   │-- 000342_mpii_test
        │   │   │   │-- 000000.jpg
        │   │   │   │-- 000001.jpg
        │   │   │   │-- ...
        │   │   │-- ...
        │   `-- test
        │       │-- 000001_mpiinew_test
        │       │   │-- 000000.jpg
        │       │   │-- 000001.jpg
        │       │   │-- ...
        │       │-- ...
        `-- mask
            │-- train
            │   │-- 000002_bonn_train
            │   │   │-- 000000.jpg
            │   │   │-- 000001.jpg
            │   │   │-- ...
            │   │-- ...
            `-- val
                │-- 000522_mpii_test
                │   │-- 000000.jpg
                │   │-- 000001.jpg
                │   │-- ...
                │-- ...
```

The official evaluation tool for PoseTrack should be installed from GitHub.

```shell
pip install git+https://github.com/svenkreiss/poseval.git
```

## sub-JHMDB dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58580-8_27">RSN (ECCV'2020)</a></summary>

```bibtex
@misc{cai2020learning,
    title={Learning Delicate Local Representations for Multi-Person Pose Estimation},
    author={Yuanhao Cai and Zhicheng Wang and Zhengxiong Luo and Binyi Yin and Angang Du and Haoqian Wang and Xinyu Zhou and Erjin Zhou and Xiangyu Zhang and Jian Sun},
    year={2020},
    eprint={2003.04030},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

</details>

For [sub-JHMDB](http://jhmdb.is.tue.mpg.de/dataset) data, please download the [images](<(http://files.is.tue.mpg.de/jhmdb/Rename_Images.tar.gz)>) from [JHMDB](http://jhmdb.is.tue.mpg.de/dataset),
Please download the annotation files from [jhmdb_annotations](https://download.openmmlab.com/mmpose/datasets/jhmdb_annotations.tar).
Move them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── jhmdb
        │-- annotations
        │   │-- Sub1_train.json
        │   |-- Sub1_test.json
        │   │-- Sub2_train.json
        │   |-- Sub2_test.json
        │   │-- Sub3_train.json
        │   |-- Sub3_test.json
        |-- Rename_Images
            │-- brush_hair
            │   │--April_09_brush_hair_u_nm_np1_ba_goo_0
            |   │   │--00001.png
            |   │   │--00002.png
            │-- catch
            │-- ...

```
