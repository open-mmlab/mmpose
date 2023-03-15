# Bottom-up Human Pose Estimation via Disentangled Keypoint Regression

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.02300">DEKR (CVPR'2021)</a></summary>

```bibtex
@inproceedings{geng2021bottom,
  title={Bottom-up human pose estimation via disentangled keypoint regression},
  author={Geng, Zigang and Sun, Ke and Xiao, Bin and Zhang, Zhaoxiang and Wang, Jingdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14676--14686},
  year={2021}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

In this paper, we are interested in the bottom-up paradigm of estimating human poses from an image. We study the dense keypoint regression framework that is previously inferior to the keypoint detection and grouping framework. Our motivation is that regressing keypoint positions accurately needs to learn representations that focus on the keypoint regions.
We present a simple yet effective approach, named disentangled keypoint regression (DEKR). We adopt adaptive convolutions through pixel-wise spatial transformer to activate the pixels in the keypoint regions and accordingly learn representations from them. We use a multi-branch structure for separate regression: each branch learns a representation with dedicated adaptive convolutions and regresses one keypoint. The resulting disentangled representations are able to attend to the keypoint regions, respectively, and thus the keypoint regression is spatially more accurate. We empirically show that the proposed direct regression method outperforms keypoint detection and grouping methods and achieves superior bottom-up pose estimation results on two benchmark datasets, COCO and CrowdPose. The code and models are available at [this https URL](https://github.com/HRNet/DEKR).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/26127467/192512411-1f8e66c1-4ba2-4db6-a66a-31c0f7c13882.png">
</div>
