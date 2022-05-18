<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html">SimpleBaseline2D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.html">CMU Panoptic HandDB (CVPR'2017)</a></summary>

```bibtex
@inproceedings{simon2017hand,
  title={Hand keypoint detection in single images using multiview bootstrapping},
  author={Simon, Tomas and Joo, Hanbyul and Matthews, Iain and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={1145--1153},
  year={2017}
}
```

</details>

Results on CMU Panoptic (MPII+NZSL val set)

| Arch                                                       | Input Size | PCKh@0.7 |  AUC  | EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :------: | :---: | :--: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_resnet_50](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/panoptic2d/res50_panoptic_256x256.py) |  256x256   |  0.999   | 0.713 | 9.00 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_panoptic_256x256-4eafc561_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_panoptic_256x256_20210330.log.json) |
