# Effective Whole-body Pose Estimation with Two-stages Distillation

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2307.15880">RTMPose (arXiv'2023)</a></summary>

```bibtex
@article{yang2023effective,
  title={Effective Whole-body Pose Estimation with Two-stages Distillation},
  author={Yang, Zhendong and Zeng, Ailing and Yuan, Chun and Li, Yu},
  journal={arXiv preprint arXiv:2307.15880},
  year={2023}
}

```

</details>

## Abstract

<!-- [ABSTRACT] -->

Whole-body pose estimation localizes the human body, hand, face, and foot keypoints in an image. This task is challenging due to multi-scale body parts, fine-grained localization for low-resolution regions, and data scarcity. Meanwhile, applying a highly efficient and accurate pose estimator to widely human-centric understanding and generation tasks is urgent. In this work, we present a two-stage pose **D**istillation for **W**hole-body **P**ose estimators, named **DWPose**, to improve their effectiveness and efficiency. The first-stage distillation designs a weight-decay strategy while utilizing a teacher's intermediate feature and final logits with both visible and invisible keypoints to supervise the student from scratch. The second stage distills the student model itself to further improve performance. Different from the previous self-knowledge distillation, this stage finetunes the student's head with only 20% training time as a plug-and-play training strategy. For data limitations, we explore the UBody dataset that contains diverse facial expressions and hand gestures for real-life applications. Comprehensive experiments show the superiority of our proposed simple yet effective methods. We achieve new state-of-the-art performance on COCO-WholeBody, significantly boosting the whole-body AP of RTMPose-l from 64.8% to 66.5%, even surpassing RTMPose-x teacher with 65.3% AP. We release a series of models with different sizes, from tiny to large, for satisfying various downstream tasks. Our code and models are available at https://github.com/IDEA-Research/DWPose.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/54797851/262253853-5f820730-260b-4685-b6c0-1a9257fb6265.jpg">
</div>
