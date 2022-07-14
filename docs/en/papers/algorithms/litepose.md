# Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Lite_Pose_Efficient_Architecture_Design_for_2D_Human_Pose_Estimation_CVPR_2022_paper.html">LitePose (CVPR'2022)</a></summary>

```bibtex
@inproceedings{wang2022lite,
  title={Lite pose: Efficient architecture design for 2d human pose estimation},
  author={Wang, Yihan and Li, Muyang and Cai, Han and Chen, Wei-Ming and Han, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13126--13136},
  year={2022}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Pose estimation plays a critical role in human-centered vision applications. However, it is difficult to deploy state-of-the-art HRNet-based pose estimation models on resource-constrained edge devices due to the high computational cost (more than 150 GMACs per frame). In this paper, we study efficient architecture design for real-time multi-person pose estimation on edge. We reveal that HRNet's high-resolution branches are redundant for models at the low-computation region via our gradual shrinking experiments. Removing them improves both efficiency and performance. Inspired by this finding, we design LitePose, an efficient single-branch architecture for pose estimation, and introduce two simple approaches to enhance the capacity of LitePose, including fusion deconv head and large kernel conv. On mobile platforms, LitePose reduces the latency by up to 5.0x without sacrificing performance, compared with prior state-of-the-art efficient pose estimation models, pushing the frontier of real-time multi-person pose estimation on edge. Our code and pre-trained models are released at https://github.com/mit-han-lab/litepose.
