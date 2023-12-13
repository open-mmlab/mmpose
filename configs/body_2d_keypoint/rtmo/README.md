# RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2312.07526">RTMO</a></summary>

```bibtex
@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

RTMO is a one-stage pose estimation model that seamlessly integrates coordinate classification into the YOLO architecture. It introduces a Dynamic Coordinate Classifier (DCC) module that handles keypoint localization through dual 1D heatmaps. The DCC employs dynamic bin allocation, localizing the coordinate bins to each predicted bounding box to improve efficiency. It also uses learnable bin representations based on positional encodings, enabling computation of bin-keypoint similarity for precise localization.

RTMO is trained end-to-end using a multi-task loss, with losses for bounding box regression, keypoint heatmap classification via a novel MLE loss, keypoint coordinate proxy regression, and keypoint visibility classification. The MLE loss models annotation uncertainty and balances optimization between easy and hard samples.

During inference, RTMO employs grid-based dense predictions to simultaneously output human detection boxes and poses in a single pass. It selectively decodes heatmaps only for high-scoring grids after NMS, minimizing computational cost.

Compared to prior one-stage methods that regress keypoint coordinates directly, RTMO achieves higher accuracy through coordinate classification while retaining real-time speeds. It also outperforms lightweight top-down approaches for images with many people, as the latter have inference times that scale linearly with the number of human instances.
