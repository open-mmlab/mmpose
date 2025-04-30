# Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Feng_Wing_Loss_for_CVPR_2018_paper.html">Wingloss (CVPR'2018)</a></summary>

```bibtex
@inproceedings{feng2018wing,
  title={Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks},
  author={Feng, Zhen-Hua and Kittler, Josef and Awais, Muhammad and Huber, Patrik and Wu, Xiao-Jun},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018},
  pages ={2235-2245},
  organization={IEEE}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

We present a new loss function, namely Wing loss, for robust facial landmark localisation with Convolutional Neural Networks (CNNs). We first compare and analyse different loss functions including L2, L1 and smooth L1. The analysis of these loss functions suggests that, for the training of a CNN-based localisation model, more attention should be paid to small and medium range errors. To this end, we design a piece-wise loss function. The new loss amplifies the impact of errors from the interval (-w, w) by switching from L1 loss to a modified logarithm function. To address the problem of under-representation of samples with large out-of-plane head rotations in the training set, we propose a simple but effective boosting strategy, referred to as pose-based data balancing. In particular, we deal with the data imbalance problem by duplicating the minority training samples and perturbing them by injecting random image rotation, bounding box translation and other data augmentation approaches. Last, the proposed approach is extended to create a two-stage framework for robust facial landmark localisation. The experimental results obtained on AFLW and 300W demonstrate the merits of the Wing loss function, and prove the superiority of the proposed method over the state-of-the-art approaches.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146528473-f3228f69-b60e-4807-9c4b-b45997fbc530.png">
</div>
