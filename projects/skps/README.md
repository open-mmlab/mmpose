# Simple Keypoints

It is a simple keypoints detector model. The model predict a heatmap and a offside map.
The result in wflw achieves 3.94 NME.

## Description

Author： @610265158.

This project implements a top-down pose estimator with custom head and loss functions that have been seamlessly inherited from existing modules within MMPose.

## Usage

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim) v0.33 or higher
- [MMPose](https://github.com/open-mmlab/mmpose) v1.0.0rc0 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `example_project/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the COCO dataset according to the [instruction](https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_body_keypoint.html#coco).

### Training commands

**To train with single GPU:**

```shell
mim train mmpose configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py
```

**To train with multiple GPUs:**

```shell
mim train mmpose configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```shell
mim train mmpose configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```shell
mim test mmpose configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py $CHECKPOINT
```

**To test with multiple GPUs:**

```shell
mim test mmpose configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```shell
mim test mmpose configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py $CHECKPOINT --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results

| Arch                                                              | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |    ckpt    |    log    |
|:------------------------------------------------------------------| :--------: |:--------------------:|:--------------------:|:----------------------------:|:-------------------------:|:--------------------:|:----------------------:|:--------------------------:| :--------: | :-------: |
| [skps](/configs/td-hm_hrnetv2-w18_skps-8xb64-60e_wflw-256x256.py) |  256x256   |         3.94         |         6.71         |             3.84             |           4.68            |         4.52         |          3.77          |            4.18            | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark-3f8e0c2c_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark_20210125.log.json) |

## Citation

> You may remove this section if not applicable.

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
