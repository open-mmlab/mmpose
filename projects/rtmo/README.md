# RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/cc2b3657-ba2b-478a-b61d-2c0fb310515c" style="width:350px" /> <img src="https://github.com/open-mmlab/mmpose/assets/26127467/94db75b7-3215-45b0-9f7e-203f31fcb263" style="width:350px" />

RTMO is a one-stage pose estimation model that achieves performance comparable to RTMPose. It has the following key advantages:

- **Faster inference speed when multiple people are present** - RTMO runs faster than RTMPose on images with more than 4 persons. This makes it well-suited for real-time multi-person pose estimation.
- **No dependency on human detectors** - Since RTMO is a one-stage model, it does not rely on an auxiliary human detector. This simplifies the pipeline and deployment.

👉🏼 TRY RTMO NOW

```bash
python demo/inferencer_demo.py $IMAGE --pose2d rtmo --vis-out-dir vis_results
```

## 📜 Introduction

Real-time multi-person pose estimation presents significant challenges in balancing speed and precision. While two-stage top-down methods slow down as the number of people in the image increases, existing one-stage methods often fail to simultaneously deliver high accuracy and real-time performance. This paper introduces RTMO, a one-stage pose estimation framework that seamlessly integrates coordinate classification by representing keypoints using dual 1-D heatmaps within the YOLO architecture, achieving accuracy comparable to top-down methods while maintaining high speed. We propose a dynamic coordinate classifier and a tailored loss function for heatmap learning, specifically designed to address the incompatibilities between coordinate classification and dense prediction models. RTMO outperforms state-of-the-art one-stage pose estimators, achieving 1.1% higher AP on COCO while operating about 9 times faster with the same backbone. Our largest model, RTMO-l, attains 74.8% AP on COCO val2017 and 141 FPS on a single V100 GPU, demonstrating its efficiency and accuracy.

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/ad94c097-7d51-4b91-b885-d8605e22a0e6" height="360px" alt><br>

Refer to [our paper](https://arxiv.org/abs/2312.07526) for more details.

## 🎉 News

- **`2023/12/13`**: The RTMO paper and models are released!

## 🗂️ Model Zoo

### Results on COCO val2017 dataset

| Model                                                        | Train Set | Latency (ms) |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                             Download                             |
| :----------------------------------------------------------- | :-------: | :----------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------------------------------------------------------------: |
| [RTMO-s](/configs/body_2d_keypoint/rtmo/coco/rtmo-s_8xb32-600e_coco-640x640.py) |   COCO    |     8.9      | 0.677 |      0.878      |      0.737      | 0.715 |      0.908      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth) |
| [RTMO-m](/configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py) |   COCO    |     12.4     | 0.709 |      0.890      |      0.778      | 0.747 |      0.920      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/coco/rtmo-l_16xb16-600e_coco-640x640.py) |   COCO    |     19.1     | 0.724 |      0.899      |      0.788      | 0.762 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_coco-640x640-516a421f_20231211.pth) |
| [RTMO-t](/configs/body_2d_keypoint/rtmo/body7/rtmo-t_8xb32-600e_body7-416x416.py) |   body7   |      -       | 0.574 |      0.803      |      0.613      | 0.611 |      0.836      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-t_8xb32-600e_body7-416x416-f48f75cb_20231219.pth) \| [onnx](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-t_8xb32-600e_body7-416x416-f48f75cb_20231219.zip) |
| [RTMO-s](/configs/body_2d_keypoint/rtmo/body7/rtmo-s_8xb32-600e_body7-640x640.py) |   body7   |     8.9      | 0.686 |      0.879      |      0.744      | 0.723 |      0.908      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth) \| [onnx](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip) |
| [RTMO-m](/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py) |   body7   |     12.4     | 0.726 |      0.899      |      0.790      | 0.763 |      0.926      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth)  \| [onnx](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py) |   body7   |     19.1     | 0.748 |      0.911      |      0.813      | 0.786 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth) \| [onnx](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip) |

- The latency is evaluated on a single V100 GPU with ONNXRuntime backend.
- "body7" refers to a combined dataset composed of [AI Challenger](https://github.com/AIChallenger/AI_Challenger_2017), [COCO](http://cocodataset.org/), [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose), [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody/), [MPII](http://human-pose.mpi-inf.mpg.de/), [PoseTrack18](https://posetrack.net/users/download.php) and [sub-JHMDB](http://jhmdb.is.tue.mpg.de/dataset).

### Results on CrowdPose test dataset

| Model                                                               | Train Set |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) |                                Download                                 |
| :------------------------------------------------------------------ | :-------: | :---: | :-------------: | :-------------: | :----: | :----: | :----: | :---------------------------------------------------------------------: |
| [RTMO-s](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-s_8xb32-700e_crowdpose-640x640.py) | CrowdPose | 0.673 |      0.882      |      0.729      | 0.737  | 0.682  | 0.591  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-700e_crowdpose-640x640-79f81c0d_20231211.pth) |
| [RTMO-m](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-m_16xb16-700e_crowdpose-640x640.py) | CrowdPose | 0.711 |      0.897      |      0.771      | 0.774  | 0.719  | 0.634  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rrtmo-m_16xb16-700e_crowdpose-640x640-0eaf670d_20231211.pth) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_crowdpose-640x640.py) | CrowdPose | 0.732 |      0.907      |      0.793      | 0.792  | 0.741  | 0.653  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_crowdpose-640x640-1008211f_20231211.pth) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py) |   body7   | 0.838 |      0.947      |      0.893      | 0.888  | 0.847  | 0.772  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth) |

## 🖥️ Train and Evaluation

### Dataset Preparation

Please follow [this instruction](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html) to prepare the training and testing datasets.

### Train

Under the root directory of mmpose, run the following command to train models:

```sh
sh tools/dist_train.sh $CONFIG $NUM_GPUS --work-dir $WORK_DIR --amp
```

- Automatic Mixed Precision (AMP) technique is used to reduce GPU memory consumption during training.

### Evaluation

Under the root directory of mmpose, run the following command to evaluate models:

```sh
sh tools/dist_test.sh $CONFIG $PATH_TO_CHECKPOINT $NUM_GPUS
```

See [here](https://mmpose.readthedocs.io/en/latest/user_guides/train_and_test.html) for more training and evaluation details.

## 🛞 Deployment

[MMDeploy](https://github.com/open-mmlab/mmdeploy) provides tools for easy deployment of RTMO models. [\[Install Now\]](https://mmdeploy.readthedocs.io/en/latest/get_started.html#installation)

**⭕ Notice**:

- PyTorch **1.12+** is required to export the ONNX model of RTMO!

- MMDeploy v1.3.1+ is required to deploy RTMO.

### ONNX Model Export

Under mmdeploy root, run:

```sh
python tools/deploy.py \
    configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic-640x640.py \
    $RTMO_CONFIG $RTMO_CHECKPOINT \
    $MMPOSE_ROOT/tests/data/coco/000000197388.jpg \
    --work-dir $WORK_DIR --dump-info \
    [--show] [--device $DEVICE]
```

### TensorRT Model Export

[Install TensorRT](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/tensorrt.html#install-tensorrt) and [build custom ops](https://mmdeploy.readthedocs.io/en/latest/05-supported-backends/tensorrt.html#build-custom-ops) first.

Then under mmdeploy root, run:

```sh
python tools/deploy.py \
    configs/mmpose/pose-detection_rtmo_tensorrt-fp16_dynamic-640x640.py \
    $RTMO_CONFIG $RTMO_CHECKPOINT \
    $MMPOSE_ROOT/tests/data/coco/000000197388.jpg \
    --work-dir $WORK_DIR --dump-info \
    --device cuda:0 [--show]
```

This conversion takes several minutes. GPU is required for TensorRT model exportation.

## ⭐ Citation

If this project benefits your work, please kindly consider citing the original paper and MMPose:

```bibtex
@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
