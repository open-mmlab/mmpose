# Bottom-up Human Pose Estimation via Disentangled Keypoint Regression (DEKR)

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

DEKR is a popular 2D bottom-up pose estimation approach that simultaneously detects all the instances and regresses the offsets from the instance centers to joints.

In order to predict the offsets more accurately, the offsets of different joints are regressed using separated branches with deformable convolutional layers. Thus convolution kernels with different shapes are adopted to extract features for the corresponding joint.
