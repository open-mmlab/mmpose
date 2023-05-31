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

Testing results on Human3.6M dataset with ground truth 2D detections

| Arch                                                                                              | MPJPE | P-MPJPE | N-MPJPE |    ckpt    |    log    |
| :------------------------------------------------------------------------------------------------ | :---: | :-----: | :-----: | :--------: | :-------: |
| [MotionBERT](/configs/body_3d_keypoint/video_pose_lift/h36m/vid_pl_motionbert_8xb32-120e_h36m.py) | 34.87 |  14.95  |  34.02  | [ckpt](<>) | [log](<>) |
