# YOLOX-Pose

This project implements a YOLOX-based human pose estimator, utilizing the approach outlined in **YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss** (CVPRW 2022). This pose estimator is lightweight and quick, making it well-suited for crowded scenes.

<img src="https://user-images.githubusercontent.com/26127467/226655503-3cee746e-6e42-40be-82ae-6e7cae2a4c7e.jpg" alt><br>

## Usage

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.6 or higher
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.6.0 or higher
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4 or higher
- [MMDetection](https://github.com/open-mmlab/mmdetection) v3.0.0rc6 or higher
- [MMYOLO](https://github.com/open-mmlab/mmyolo) v0.5.0 or higher
- [MMPose](https://github.com/open-mmlab/mmpose) v1.0.0rc1 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. **In `yolox-pose/` root directory**, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Inference

Users can apply YOLOX-Pose models to estimate human poses using the inferencer found in the MMPose core package. Use the command below:

```shell
python demo/inferencer_demo.py $INPUTS \
    --pose2d $CONFIG --pose2d-weights $CHECKPOINT --scope mmyolo \
    [--show] [--vis-out-dir $VIS_OUT_DIR] [--pred-out-dir $PRED_OUT_DIR]
```

For more information on using the inferencer, please see [this document](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html#out-of-the-box-inferencer).

Here's an example code:

```shell
python demo/inferencer_demo.py ../../tests/data/coco/000000000785.jpg \
    --pose2d configs/yolox-pose_s_8xb32-300e_coco.py \
    --pose2d-weights https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_s_8xb32-300e_coco-9f5e3924_20230321.pth \
    --scope mmyolo --vis-out-dir vis_results
```

This will create an output image `vis_results/000000000785.jpg`, which appears like:

<img src="https://user-images.githubusercontent.com/26127467/226552585-19b91294-9751-4599-98e7-5dae071a1761.jpg" height="360px" alt><br>

### Training & Testing

#### Data Preparation

Prepare the COCO dataset according to the [instruction](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#coco).

#### Commands

**To train with multiple GPUs:**

```shell
bash tools/dist_train.sh $CONFIG 8 --amp
```

**To train with slurm:**

```shell
bash tools/slurm_train.sh $PARTITION $JOBNAME $CONFIG $WORKDIR --amp
```

**To test with single GPU:**

```shell
python tools/test.py $CONFIG $CHECKPOINT
```

**To test with multiple GPUs:**

```shell
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8
```

**To test with multiple GPUs by slurm:**

```shell
bash tools/slurm_test.sh $PARTITION $JOBNAME $CONFIG $CHECKPOINT
```

### Results

Results on COCO val2017

|                              Model                              | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                                 Download                                 |
| :-------------------------------------------------------------: | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :----------------------------------------------------------------------: |
| [YOLOX-tiny-Pose](./configs/yolox-pose_tiny_4xb64-300e_coco.py) |    416     | 0.518 |      0.799      |      0.545      | 0.566 |      0.841      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_tiny_4xb64-300e_coco-c47dd83b_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_tiny_4xb64-300e_coco_20230321.json) |
|    [YOLOX-s-Pose](./configs/yolox-pose_s_8xb32-300e_coco.py)    |    640     | 0.632 |      0.875      |      0.692      | 0.676 |      0.907      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_s_8xb32-300e_coco-9f5e3924_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_s_8xb32-300e_coco_20230321.json) |
|    [YOLOX-m-Pose](./configs/yolox-pose_m_4xb64-300e_coco.py)    |    640     | 0.685 |      0.897      |      0.753      | 0.727 |      0.925      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_m_4xb64-300e_coco-cbd11d30_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_m_4xb64-300e_coco_20230321.json) |
|    [YOLOX-l-Pose](./configs/yolox-pose_l_4xb64-300e_coco.py)    |    640     | 0.706 |      0.907      |      0.775      | 0.747 |      0.934      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_l_4xb64-300e_coco-122e4cf8_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_l_4xb64-300e_coco_20230321.json) |

We have only trained models with an input size of 640, as we couldn't replicate the performance enhancement mentioned in the paper when increasing the input size from 640 to 960. We warmly welcome any contributions if you can successfully reproduce the results from the paper!

**NEW!**

[MMYOLO](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolox/README.md#yolox-pose) also supports YOLOX-Pose and achieves better performance. Their models are fully compatible with this project. Here are their results on COCO val2017:

|  Backbone  | Size | Batch Size | AMP | RTMDet-Hyp | Mem (GB) |  AP  |                                   Config                                   |                                   Download                                    |
| :--------: | :--: | :--------: | :-: | :--------: | :------: | :--: | :------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| YOLOX-tiny | 416  |   8xb32    | Yes |    Yes     |   5.3    | 52.8 | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolox/pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco_20230427_080351-2117af67.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco/yolox-pose_tiny_8xb32-300e-rtmdet-hyp_coco_20230427_080351.log.json) |
|  YOLOX-s   | 640  |   8xb32    | Yes |    Yes     |   10.7   | 63.7 | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolox/pose/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco_20230427_005150-e87d843a.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco/yolox-pose_s_8xb32-300e-rtmdet-hyp_coco_20230427_005150.log.json) |
|  YOLOX-m   | 640  |   8xb32    | Yes |    Yes     |   19.2   | 69.3 | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolox/pose/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco_20230427_094024-bbeacc1c.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco/yolox-pose_m_8xb32-300e-rtmdet-hyp_coco_20230427_094024.log.json) |
|  YOLOX-l   | 640  |   8xb32    | Yes |    Yes     |   30.3   | 71.1 | [config](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolox/pose/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco.py) | [model](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco_20230427_041140-82d65ac8.pth) \| [log](https://download.openmmlab.com/mmyolo/v0/yolox/pose/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco/yolox-pose_l_8xb32-300e-rtmdet-hyp_coco_20230427_041140.log.json) |

## Citation

If this project benefits your work, please kindly consider citing the original papers:

```bibtex
@inproceedings{maji2022yolo,
  title={YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss},
  author={Maji, Debapriya and Nagori, Soyeb and Mathew, Manu and Poddar, Deepak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
}
```

```bibtex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

Additionally, please cite our work as well:

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
