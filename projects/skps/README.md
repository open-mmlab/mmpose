# Simple Keypoints

## Description

Author： @2120140200@mail.nankai.edu.cn

It is a simple keypoints detector model. The model predict a score heatmap and an encoded location map.
The result in wflw achieves 3.94 NME.

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
mim train mmpose configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py
```

**To train with multiple GPUs:**

```shell
mim train mmpose configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```shell
mim train mmpose configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```shell
mim test mmpose configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py -C $CHECKPOINT
```

**To test with multiple GPUs:**

```shell
mim test mmpose configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py -C $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```shell
mim test mmpose configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py -C $CHECKPOINT --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results

WFLW

| Arch       | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |    ckpt    |    log    |
| :--------- | :--------: | :------------------: | :------------------: | :--------------------------: | :-----------------------: | :------------------: | :--------------------: | :------------------------: | :--------: | :-------: |
| [skps](/configs/td-hm_hrnetv2-w18_skps-1xb64-80e_wflw-256x256.py) |  256x256   |         3.88         |         6.60         |             3.81             |           4.57            |         4.44         |          3.75          |            4.13            | [ckpt](https://drive.google.com/file/d/10U7f_pp0BPMhm575cO7Vg13nZVCy-PfN/view?usp=sharing) | [log](https://drive.google.com/file/d/1fBNcAyMdRr9nTN8wHqvYPnd9AFBinOk4/view?usp=sharing) |
