<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2210.06551">MotionBERT (2022)</a></summary>

```bibtex
 @misc{Zhu_Ma_Liu_Liu_Wu_Wang_2022,
 title={Learning Human Motion Representations: A Unified Perspective},
 author={Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
 year={2022},
 month={Oct},
 language={en-US}
 }
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
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

Results on Human3.6M dataset with ground truth 2D detections

| Arch                                                                                    | MPJPE | average MPJPE | P-MPJPE |                                           ckpt                                           |
| :-------------------------------------------------------------------------------------- | :---: | :-----------: | :-----: | :--------------------------------------------------------------------------------------: |
| [MotionBERT\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m.py) | 34.5  |     34.6      |  27.1   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth) |
| [MotionBERT-finetuned\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py) | 26.9  |     26.8      |  21.0   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth) |

Results on Human3.6M dataset converted from the [official repo](https://github.com/Walter0807/MotionBERT)<sup>1</sup> with ground truth 2D detections

| Arch                                                                                   | MPJPE | average MPJPE | P-MPJPE |                                          ckpt                                          | log |
| :------------------------------------------------------------------------------------- | :---: | :-----------: | :-----: | :------------------------------------------------------------------------------------: | :-: |
| [MotionBERT\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m-original.py) | 39.8  |     39.2      |  33.4   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth) |  /  |
| [MotionBERT-finetuned\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m-original.py) | 37.7  |     37.2      |  32.2   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth) |  /  |

<sup>1</sup> By default, we test models with [Human 3.6m dataset](/docs/en/dataset_zoo/3d_body_keypoint.md#human3-6m) processed by MMPose. The official repo's dataset includes more data and applies a different pre-processing technique. To achieve the same result with the official repo, please download the [test annotation file](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/h36m_test_original.npz), [train annotation file](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/h36m_train_original.npz) and [factors](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/h36m_factors.npy) under `$MMPOSE/data/h36m/annotation_body3d/fps50` and test with the configs we provided.

*Models with * are converted from the [official repo](https://github.com/Walter0807/MotionBERT). The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*
