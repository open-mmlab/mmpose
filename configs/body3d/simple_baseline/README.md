# A simple yet effective baseline for 3d human pose estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

## Results and models

### 3D Human Pose Estimation

#### Results on Human3.6M dataset with ground truth 2D detections

| Arch | MPJPE | P-MPJPE | ckpt | log |
| :--- | :---: | :---: | :---: | :---: |
| [simple3Dbaseline<sup>1</sup>](/configs/body3d/simple_baseline/h36m/simple3Dbaseline_h36m.py) | 43.4 | 34.3 | [ckpt](https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth) | [log](https://download.openmmlab.com/mmpose/body3d/simple_baseline/20210415_065056.log.json) |

<sup>1</sup> Differing from the original paper, we didn't apply the `max-norm constraint` because we found this led to a better convergence and performance.
