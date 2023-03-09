# Learning Temporal Pose Estimation from Sparsely-Labeled Videos

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1906.04016">PoseWarper (NeurIPS'2019)</a></summary>

```bibtex
@inproceedings{NIPS2019_gberta,
title = {Learning Temporal Pose Estimation from Sparsely Labeled Videos},
author = {Bertasius, Gedas and Feichtenhofer, Christoph, and Tran, Du and Shi, Jianbo, and Torresani, Lorenzo},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2019},
}
```

</details>

PoseWarper proposes a network that leverages training videos with sparse annotations (every k frames) to learn to perform dense temporal pose propagation and estimation. Given a pair of video frames, a labeled Frame A and an unlabeled Frame B, the model is trained to predict human pose in Frame A using the features from Frame B by means of deformable convolutions to implicitly learn the pose warping between A and B.

The training of PoseWarper can be split into two stages.

The first-stage is trained with the pre-trained model and the main backbone is fine-tuned in a single-frame setting.

The second-stage is trained with the model from the first stage, and the warping offsets are learned in a multi-frame setting while the backbone is frozen.
