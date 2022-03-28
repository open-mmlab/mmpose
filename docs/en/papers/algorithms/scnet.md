# Improving Convolutional Networks with Self-Calibrated Convolutions

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Improving_Convolutional_Networks_With_Self-Calibrated_Convolutions_CVPR_2020_paper.html">SCNet (CVPR'2020)</a></summary>

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
  year={2020}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Recent advances on CNNs are mostly devoted to designing more complex architectures to enhance their representation learning capacity. In this paper, we consider how to improve the basic convolutional feature transformation process of CNNs without tuning the model architectures. To this end, we present a novel self-calibrated convolutions that explicitly expand fields-of-view of each convolutional layers through internal communications and hence enrich the output features. In particular, unlike the standard convolutions that fuse spatial and channel-wise information using small kernels (e.g., 3x3), self-calibrated convolutions adaptively build long-range spatial and inter-channel dependencies around each spatial location through a novel self-calibration operation. Thus, it can help CNNs generate more discriminative representations by explicitly incorporating richer information. Our self-calibrated convolution design is simple and generic, and can be easily applied to augment standard convolutional layers without introducing extra parameters and complexity. Extensive experiments demonstrate that when applying self-calibrated convolutions into different backbones, our networks can significantly improve the baseline models in a variety of vision tasks, including image recognition, object detection, instance segmentation, and keypoint detection, with no need to change the network architectures. We hope this work could provide a promising way for future research in designing novel convolutional feature transformations for improving convolutional networks. Code is available on the project page.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522623-fdc88996-1a8f-4b0a-bdee-fc284dfd7348.png">
</div>
