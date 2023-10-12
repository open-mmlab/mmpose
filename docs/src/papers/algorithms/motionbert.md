# MotionBERT: Unified Pretraining for Human Motion Analysis

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2210.06551">MotionBERT (ICCV'2023)</a></summary>

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

## Abstract

<!-- [ABSTRACT] -->

We present MotionBERT, a unified pretraining framework, to tackle different sub-tasks of human motion analysis including 3D pose estimation, skeleton-based action recognition, and mesh recovery. The proposed framework is capable of utilizing all kinds of human motion data resources, including motion capture data and in-the-wild videos. During pretraining, the pretext task requires the motion encoder to recover the underlying 3D motion from noisy partial 2D observations. The pretrained motion representation thus acquires geometric, kinematic, and physical knowledge about human motion and therefore can be easily transferred to multiple downstream tasks. We implement the motion encoder with a novel Dual-stream Spatio-temporal Transformer (DSTformer) neural network. It could capture long-range spatio-temporal relationships among the skeletal joints comprehensively and adaptively, exemplified by the lowest 3D pose estimation error so far when trained from scratch. More importantly, the proposed framework achieves state-of-the-art performance on all three downstream tasks by simply finetuning the pretrained motion encoder with 1-2 linear layers, which demonstrates the versatility of the learned motion representations.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmpose/assets/13503330/877d47ee-b821-476c-a805-f39ca656913c">
</div>
