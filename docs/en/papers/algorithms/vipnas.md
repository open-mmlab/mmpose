# ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2105.10154">ViPNAS (CVPR'2021)</a></summary>

```bibtex
@article{xu2021vipnas,
  title={ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search},
  author={Xu, Lumin and Guan, Yingda and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Human pose estimation has achieved significant progress in recent years. However, most of the recent methods focus on improving accuracy using complicated models and ignoring real-time efficiency. To achieve a better trade-off between accuracy and efficiency, we propose a novel neural architecture search (NAS) method, termed ViPNAS, to search networks in both spatial and temporal levels for fast online video pose estimation. In the spatial level, we carefully design the search space with five different dimensions including network depth, width, kernel size, group number, and attentions. In the temporal level, we search from a series of temporal feature fusions to optimize the total accuracy and speed across multiple video frames. To the best of our knowledge, we are the first to search for the temporal feature fusion and automatic computation allocation in videos. Extensive experiments demonstrate the effectiveness of our approach on the challenging COCO2017 and PoseTrack2018 datasets. Our discovered model family, S-ViPNAS and T-ViPNAS, achieve significantly higher inference speed (CPU real-time) without sacrificing the accuracy compared to the previous state-of-the-art methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146527709-b8e55e40-426c-4958-8a35-b8d259e69efa.png">
</div>
