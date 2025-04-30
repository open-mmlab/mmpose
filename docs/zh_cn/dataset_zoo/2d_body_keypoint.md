# 2D 人体关键点数据集

建议将数据集的根目录链接到 `$MMPOSE/data`。
如果你的文件夹结构不同，你可能需要在配置文件中更改相应的路径。

MMPose 支持的数据集：

- Images
  - [COCO](#coco) \[ [主页](http://cocodataset.org/) \]
  - [MPII](#mpii) \[ [主页](http://human-pose.mpi-inf.mpg.de/) \]
  - [MPII-TRB](#mpii-trb) \[ [主页](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) \]
  - [AI Challenger](#aic) \[ [主页](https://github.com/AIChallenger/AI_Challenger_2017) \]
  - [CrowdPose](#crowdpose) \[ [主页](https://github.com/Jeff-sjtu/CrowdPose) \]
  - [OCHuman](#ochuman) \[ [主页](https://github.com/liruilong940607/OCHumanApi) \]
  - [MHP](#mhp) \[ [主页](https://lv-mhp.github.io/dataset) \]
  - [Human-Art](#humanart) \[ [主页](https://idea-research.github.io/HumanArt/) \]
  - [ExLPose](#exlpose-dataset) \[ [Homepage](http://cg.postech.ac.kr/research/ExLPose/) \]
- Videos
  - [PoseTrack18](#posetrack18) \[ [主页](https://posetrack.net/users/download.php) \]
  - [sub-JHMDB](#sub-jhmdb-dataset) \[ [主页](http://jhmdb.is.tue.mpg.de/dataset) \]

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png" height="300px">
</div>

对于 [COCO](http://cocodataset.org/) 数据，请从 [COCO 下载](http://cocodataset.org/#download) 中下载，需要 2017 训练/验证集进行 COCO 关键点的训练和验证。
[HRNet-Human-Pose-Estimation](https://github.com/HRNet/HRNet-Human-Pose-Estimation) 提供了 COCO val2017 的人体检测结果，以便重现我们的多人姿态估计结果。
请从 [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) 或 [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing) 下载。
如果要在 COCO'2017 test-dev 上进行评估，请下载 [image-info](https://download.openmmlab.com/mmpose/datasets/person_keypoints_test-dev-2017.json)。

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864660-e5f51e7d-deca-41d8-9725-8b5432bcc0e6.png" height="300px">
</div>

对于 [MPII](http://human-pose.mpi-inf.mpg.de/) 数据，请从 [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/) 下载。
我们已将原始的注释文件转换为 json 格式，请从 [mpii_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar) 下载。
请将它们解压到 {MMPose}/data 下，并确保目录结构如下：

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

在训练和推理期间，默认情况下，预测结果将以 '.mat' 格式保存。我们还提供了一个工具，以将这个 '.mat' 转换为更易读的 '.json' 格式。

```shell
python tools/dataset/mat2json ${PRED_MAT_FILE} ${GT_JSON_FILE} ${OUTPUT_PRED_JSON_FILE}
```

例如，

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864382-ab722299-6806-4ae4-babb-7bcc5fb09662.png" height="300px">
</div>

对于 [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body) 数据，请从 [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/) 下载。
请从 [mpii_trb_annotations](https://download.openmmlab.com/mmpose/datasets/mpii_trb_annotations.tar) 下载注释文件。
将它们解压到 {MMPose}/data 下，并确保它们的结构如下：

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864755-dd19644e-fccb-458b-a8c0-de55920261f5.png" height="300px">
</div>

对于 [AIC](https://github.com/AIChallenger/AI_Challenger_2017) 数据，请从 [AI Challenger 2017](https://github.com/AIChallenger/AI_Challenger_2017) 下载。其中 2017 Train/Val 数据适用于关键点的训练和验证。
请从 [aic_annotations](https://download.openmmlab.com/mmpose/datasets/aic_annotations.tar) 下载注释文件。
下载并解压到 $MMPOSE/data 下，并确保它们的结构如下：

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864868-54a98493-df3a-44d8-acbc-6ec22043dfb9.png" height="300px">
</div>

对于 [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose)数据，请从 [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) 下载。
请下载标注文件和人体检测结果从 [crowdpose_annotations](https://download.openmmlab.com/mmpose/datasets/crowdpose_annotations.tar)。
对于自上而下的方法，我们按照 [CrowdPose](https://arxiv.org/abs/1812.00324)使用[YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) 的 [预训练权重](https://pjreddie.com/media/files/yolov3.weights) 来生成检测到的人体边界框。
对于模型训练，我们按照 [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) 在 CrowdPose 的训练/验证数据集上进行训练，并在 CrowdPose 测试数据集上评估模型。
下载并解压缩它们到 $MMPOSE/data 目录下，结构应如下：

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png" height="300px">
</div>

对于 [OCHuman](https://github.com/liruilong940607/OCHumanApi) 数据，请从 [OCHuman](https://github.com/liruilong940607/OCHumanApi) 下载图像和标注。
将它们移动到 $MMPOSE/data 目录下，结构应如下：

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227865030-2fd33ade-2cc2-4b67-aca0-6dea2124b63c.png" height="300px">
</div>

对于 [MHP](https://lv-mhp.github.io/dataset) 数据，请从 [MHP](https://lv-mhp.github.io/dataset) 下载。
请从 [mhp_annotations](https://download.openmmlab.com/mmpose/datasets/mhp_annotations.tar.gz) 下载标注文件。
请下载并解压到 $MMPOSE/data 目录下，并确保目录结构如下：

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

## Human-Art dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://idea-research.github.io/HumanArt/">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023humanart,
    title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
    author={Ju, Xuan and Zeng, Ailing and Jianan, Wang and Qiang, Xu and Lei, Zhang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    year={2023}}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png" height="300px">
</div>

对于 [Human-Art](https://idea-research.github.io/HumanArt/) 数据，请从 [其网站](https://idea-research.github.io/HumanArt/) 下载图像和标注文件。
您需要填写 [申请表](https://docs.google.com/forms/d/e/1FAIpQLScroT_jvw6B9U2Qca1_cl5Kmmu1ceKtlh6DJNmWLte8xNEhEw/viewform) 以获取数据访问权限。
请将它们移动到 $MMPOSE/data 目录下，目录结构应如下：

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
|── data
    │── HumanArt
        │-- images
        │   │-- 2D_virtual_human
        │   │   |-- cartoon
        │   │   |   |-- 000000000000.jpg
        │   │   |   |-- ...
        │   │   |-- digital_art
        │   │   |-- ...
        │   |-- 3D_virtual_human
        │   |-- real_human
        |-- annotations
        │   │-- validation_humanart.json
        │   │-- training_humanart_coco.json
        |-- person_detection_results
        │   │-- HumanArt_validation_detections_AP_H_56_person.json
```

您可以选择是否下载 Human-Art 的其他标注文件。如果你想使用其他标注文件（例如，卡通的验证集），你需要在配置文件中编辑相应的代码。

## ExLPose dataset

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://cg.postech.ac.kr/research/ExLPose/">ExLPose (2023)</a></summary>

```bibtex
@inproceedings{ExLPose_2023_CVPR,
 title={Human Pose Estimation in Extremely Low-Light Conditions},
 author={Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim, ByungJu Woo, Haechan Lee, Sunghyun Cho, Suha Kwak},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2023}
}
```

</details>

<div align="center">
  <img src="https://github.com/open-mmlab/mmpose/assets/71805205/d2c7d552-249a-4ac0-8ac3-1467ace59f2f" height="300px">
</div>

请从 [ExLPose](https://drive.google.com/drive/folders/1E0Is4_cShxvsbJlep_aNEYLJpmHzq9FL) 下载数据，将其移动到 $MMPOSE/data 目录下，并使其结构如下：

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── ExLPose
        │-- annotations
        |	|-- ExLPose
        │   |-- ExLPose_train_LL.json
        │   |-- ExLPose_test_LL-A.json
        │   |-- ExLPose_test_LL-E.json
        │   |-- ExLPose_test_LL-H.json
        │   |-- ExLPose_test_LL-N.json
        |-- dark
            |--00001.png
            |--00002.png
            |--...

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227865114-3f98c673-f6d0-4518-ae99-653f475f9fc8.png" height="300px">
</div>

关于 [PoseTrack18](https://posetrack.net/users/download.php) 的数据，请从 [PoseTrack18](https://posetrack.net/users/download.php) 下载。
请从 [posetrack18_annotations](https://download.openmmlab.com/mmpose/datasets/posetrack18_annotations.tar) 下载标注文件。
我们已经将分散在各个视频中的官方标注文件合并为两个 json 文件（posetrack18_train & posetrack18_val.json）。我们还生成了 [mask 文件](https://download.openmmlab.com/mmpose/datasets/posetrack18_mask.tar) 以加速训练。
对于自上而下的方法，我们使用 [MMDetection](https://github.com/open-mmlab/mmdetection) 预训练的 [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth)（X-101-64x4d-FPN）来生成检测到的人体边界框。
请下载并将它们解压到 $MMPOSE/data 下，目录结构应如下所示：

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

官方的 PoseTrack 评估工具可以使用以下命令安装。

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

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227865619-d65f64ae-991d-4693-99c2-caecd1beb1fc.png" height="300px">
</div>

对于 [sub-JHMDB](http://jhmdb.is.tue.mpg.de/dataset) 的数据，请从 [JHMDB](http://jhmdb.is.tue.mpg.de/dataset) 下载 [图像](<(http://files.is.tue.mpg.de/jhmdb/Rename_Images.tar.gz)>)，
请从 [jhmdb_annotations](https://download.openmmlab.com/mmpose/datasets/jhmdb_annotations.tar) 下载标注文件。
将它们移至 $MMPOSE/data 下，并使目录结构如下所示：

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
