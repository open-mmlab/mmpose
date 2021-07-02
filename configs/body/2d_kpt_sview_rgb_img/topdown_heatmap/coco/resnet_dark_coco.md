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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50_dark](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192_dark.py) | 256x192 | 0.724 | 0.898 | 0.800 | 0.777 | 0.936 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark-43379d20_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark_20200709.log.json) |
| [pose_resnet_50_dark](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288_dark.py) | 384x288 | 0.735 | 0.900 | 0.801 | 0.785 | 0.937 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark-33d3e5e5_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark_20210203.log.json) |
| [pose_resnet_101_dark](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192_dark.py) | 256x192 | 0.732 | 0.899 | 0.808 | 0.786 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark-64d433e6_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark_20200812.log.json) |
| [pose_resnet_101_dark](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288_dark.py) | 384x288 | 0.749 | 0.902 | 0.816 | 0.799 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark-cb45c88d_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark_20210203.log.json) |
| [pose_resnet_152_dark](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192_dark.py) | 256x192 | 0.745 | 0.905 | 0.821 | 0.797 | 0.942 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark-ab4840d5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark_20200812.log.json) |
| [pose_resnet_152_dark](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288_dark.py) | 384x288 | 0.757 | 0.909 | 0.826 | 0.806 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark-d3b8ebd7_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark_20210203.log.json) |
