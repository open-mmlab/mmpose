# Numerical Coordinate Regression with Convolutional Neural Networks

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1801.07372v2">DSNT (2018)</a></summary>

```bibtex
@article{nibali2018numerical,
  title={Numerical Coordinate Regression with Convolutional Neural Networks},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1801.07372},
  year={2018}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

We study deep learning approaches to inferring numerical coordinates for points of interest in an input image. Existing convolutional neural network-based solutions to this problem either take a heatmap matching approach or regress to coordinates with a fully connected output layer. Neither of these approaches is ideal, since the former is not entirely differentiable, and the latter lacks inherent spatial generalization. We propose our differentiable spatial to numerical transform (DSNT) to fill this gap. The DSNT layer adds no trainable parameters, is fully differentiable, and exhibits good spatial generalization. Unlike heatmap matching, DSNT works well with low heatmap resolutions, so it can be dropped in as an output layer for a wide range of existing fully convolutional architectures. Consequently, DSNT offers a better trade-off between inference speed and prediction accuracy compared to existing techniques. When used to replace the popular heatmap matching approach used in almost all state-of-the-art methods for pose estimation, DSNT gives better prediction accuracy for all model architectures tested.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/189809838-1f313583-e73b-4843-b396-70b4c979ea77.png">
</div>
