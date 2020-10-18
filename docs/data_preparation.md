# Prepare datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [COCO](http://cocodataset.org/)
- [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/)
- [MPII](http://human-pose.mpi-inf.mpg.de/)
- [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
- [AI Challenger](https://github.com/AIChallenger/AI_Challenger_2017)
- [OCHuman](https://github.com/liruilong940607/OCHumanApi)
- [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)
- [OneHand10K](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html)
- [FreiHand](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
- [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html)

## COCO

For COCO data, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) provides person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-).
Download and extract them under $MMPOSE/data, and make them look like this:

```
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
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
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

## COCO-WholeBody

For COCO-WholeBody datatset, images can be downloaded from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download COCO-WholeBody annotations for COCO-WholeBody annotations for [Train](https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf/view?usp=sharing) / [Validation](https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb/view?usp=sharing) (Google Drive).
Download person detection result of COCO val2017 from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-).
Download and extract them under $MMPOSE/data, and make them look like this:

```
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- annotations
        │   │-- coco_wholebody_train_v1.0.json
        │   |-- coco_wholebody_val_v1.0.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
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
Please also install the latest version of [Extended COCO API](https://github.com/jin-s13/xtcocoapi) (version>=1.5) to support COCO-WholeBody evaluation:

```pip install xtcocotools```

## MPII

For MPII data, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
We have converted the original annotation files into json format, please download them from [mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```
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
python tools/mat2json ${PRED_MAT_FILE} ${GT_JSON_FILE} ${OUTPUT_PRED_JSON_FILE}
```
For example,
```shell
python tools/mat2json work_dirs/res50_mpii_256x256/pred.mat data/mpii/annotations/mpii_val.json pred.json
```

## MPII-TRB

For MPII-TRB data, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/).
Please download the annotation files from [mpii_trb_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_trb_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```
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

For AIC data, please download from [AI Challenger 2017](https://github.com/AIChallenger/AI_Challenger_2017), 2017 Train/Val is needed for keypoints training and validation.
Please download the annotation files from [aic_annotations](https://download.openmmlab.com/mmpose/datasets/aic_annotations.tar).
Download and extract them under $MMPOSE/data, and make them look like this:

```
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

For CrowdPose data, please download from [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose).
Please download the annotation files from [crowdpose_annotations](https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar).
For top-down approaches, we follow [CrowdPose](https://arxiv.org/abs/1812.00324) to use the [pre-trained weights](https://pjreddie.com/media/files/yolov3.weights) of [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) to generate the detected human bounding boxes.
For model training, we follow [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) to train models on CrowdPose train/val dataset, and evaluate models on CrowdPose test dataset.
Download and extract them under $MMPOSE/data, and make them look like this:

```
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

## PoseTrack18

For PoseTrack18 data, please download from [PoseTrack18](https://posetrack.net/users/download.php).
Please download the annotation files from [posetrack18_annotations](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/datasets/posetrack18_annotations.tar).
We have merged the video-wise separated official annotation files into two json files (posetrack18_train & posetrack18_val.json).
The person detection results of PoseTrack2018 val are from [LightTrack](https://github.com/Guanghan/lighttrack). To save space, we have filtered out some low-score boxes (score < 0.1).
Download and extract them under $MMPOSE/data, and make them look like this:

```
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
            │-- train
            │   │-- 000001_bonn_train
            │   │   │-- 000000.jpg
            │   │   │-- 000001.jpg
            │   │   │-- ...
            │   │-- ...
            │-- val
            │   │-- 000342_mpii_test
            │   │   │-- 000000.jpg
            │   │   │-- 000001.jpg
            │   │   │-- ...
            │   │-- ...
            `-- test
                │-- 000001_mpiinew_test
                │   │-- 000000.jpg
                │   │-- 000001.jpg
                │   │-- ...
                │-- ...
```

The official evaluation tool for PoseTrack should be installed from GitHub.
```
pip install git+https://github.com/svenkreiss/poseval.git
```


## OCHuman

For OCHuman data, please download the images and annotations from [OCHuman](https://github.com/liruilong940607/OCHumanApi),
Move them under $MMPOSE/data, and make them look like this:

```
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

## OneHand10K

For OneHand10K data, please download from [OneHand10K Dataset](https://www.yangangwang.com/papers/WANG-MCC-2018-10.html).
Please download the annotation files from [onehand10k_annotations](https://download.openmmlab.com/mmpose/datasets/onehand10k_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```
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

For FreiHAND data, please download from [FreiHand Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html).
Since the official dataset does not provide validation set, we randomly split the training data into 8:1:1 for train/val/test.
Please download the annotation files from [freihand_annotations](https://download.openmmlab.com/mmpose/datasets/frei_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```
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

For CMU Panoptic HandDB, please download from [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html).
Following [Simon et al](https://arxiv.org/abs/1704.07809), panoptic images (hand143_panopticdb) and MPII & NZSL training sets (manual_train) are used for training, while MPII & NZSL test set (manual_test) for testing.
Please download the annotation files from [panoptic_annotations](https://download.openmmlab.com/mmpose/datasets/panoptic_annotations.tar).
Extract them under {MMPose}/data, and make them look like this:

```
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
