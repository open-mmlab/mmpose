# Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2302.01593.pdf">ED-Pose (ICLR'2023)</a></summary>

```bibtex
@inproceedings{
yang2023explicit,
title={Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation},
author={Jie Yang and Ailing Zeng and Shilong Liu and Feng Li and Ruimao Zhang and Lei Zhang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=s4WVupnJjmX}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

This paper presents a novel end-to-end framework with Explicit box Detection for multi-person Pose estimation, called ED-Pose, where it unifies the contextual learning between human-level (global) and keypoint-level (local) information. Different from previous one-stage methods, ED-Pose re-considers this task as two explicit box detection processes with a unified representation and regression supervision. First, we introduce a human detection decoder from encoded tokens to extract global features. It can provide a good initialization for the latter keypoint detection, making the training process converge fast. Second, to bring in contextual information near keypoints, we regard pose estimation as a keypoint box detection problem to learn both box positions and contents for each keypoint. A human-to-keypoint detection decoder adopts an interactive learning strategy between human and keypoint features to further enhance global and local feature aggregation. In general, ED-Pose is conceptually simple without post-processing and dense heatmap supervision. It demonstrates its effectiveness and efficiency compared with both two-stage and one-stage methods. Notably, explicit box detection boosts the pose estimation performance by 4.5 AP on COCO and 9.9 AP on CrowdPose. For the first time, as a fully end-to-end framework with a L1 regression loss, ED-Pose surpasses heatmap-based Top-down methods under the same backbone by 1.2 AP on COCO and achieves the state-of-the-art with 76.6 AP on CrowdPose without bells and whistles. Code is available at https://github.com/IDEA-Research/ED-Pose.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/IDEA-Research/ED-Pose/raw/master/figs/edpose_git.jpg">
</div>
