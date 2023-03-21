# YOLOX-Pose

This project implements a YOLOX-based human pose estimator, utilizing the approach outlined in **YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss** (CVPRW 2022). This pose estimator is lightweight and quick, making it well-suited for crowded scenes.

<img src="https://user-images.githubusercontent.com/26127467/226550744-3dd948f4-cc5a-4a2f-a737-c595fc6dfe4d.jpg" alt><br>

## Usage

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.6 or higher
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.6.0 or higher
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0rc4 or higher
- [MMDetection](https://github.com/open-mmlab/mmdetection) v3.0.0rc6 or higher
- [MMYOLO](https://github.com/open-mmlab/mmyolo) v0.5.0 or higher
- [MMPose](https://github.com/open-mmlab/mmpose) v1.0.0rc1 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `yolox-pose/` root directory, run the following line to add the current directory to `PYTHONPATH`:

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

For more information on using the inferencer, please see [this document](https://mmpose.readthedocs.io/en/1.x/user_guides/inference.html#out-of-the-box-inferencer).

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

Prepare the COCO dataset according to the [instruction](https://mmpose.readthedocs.io/en/1.x/dataset_zoo/2d_body_keypoint.html#coco).

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
| [YOLOX-tiny-Pose](./configs/yolox-pose_tiny_4xb64-300e_coco.py) |    640     | 0.477 |      0.756      |      0.506      | 0.547 |      0.802      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_tiny_4xb64-300e_coco-c47dd83b_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_tiny_4xb64-300e_coco_20230321.json) |
|    [YOLOX-s-Pose](./configs/yolox-pose_s_8xb32-300e_coco.py)    |    640     | 0.595 |      0.836      |      0.653      | 0.658 |      0.878      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_s_8xb32-300e_coco-9f5e3924_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_s_8xb32-300e_coco_20230321.json) |
|    [YOLOX-m-Pose](./configs/yolox-pose_m_4xb64-300e_coco.py)    |    640     | 0.659 |      0.870      |      0.729      | 0.713 |      0.903      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_m_4xb64-300e_coco-cbd11d30_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_m_4xb64-300e_coco_20230321.json) |
|    [YOLOX-l-Pose](./configs/yolox-pose_l_4xb64-300e_coco.py)    |    640     | 0.679 |      0.882      |      0.749      | 0.733 |      0.911      | [model](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_l_4xb64-300e_coco-122e4cf8_20230321.pth) \| [log](https://download.openmmlab.com/mmpose/v1/projects/yolox-pose/yolox-pose_l_4xb64-300e_coco_20230321.json) |

We have only trained models with an input size of 640, as we couldn't replicate the performance enhancement mentioned in the paper when increasing the input size from 640 to 960. We warmly welcome any contributions if you can successfully reproduce the results from the paper!

## Citation

If this project benefits your work, please kindly consider citing the original paper:

```bibtex
@inproceedings{maji2022yolo,
  title={YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss},
  author={Maji, Debapriya and Nagori, Soyeb and Mathew, Manu and Poddar, Deepak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
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
