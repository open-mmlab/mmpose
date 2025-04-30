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

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-256x192.py) |  256x192   | 0.724 |      0.897      |      0.797      | 0.777 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-256x192-c129dcb6_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-256x192_20220926.log) |
| [pose_resnet_50_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-384x288.py) |  384x288   | 0.735 |      0.902      |      0.801      | 0.786 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-384x288-8b90b538_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_dark-8xb64-210e_coco-384x288_20220926.log) |
| [pose_resnet_101_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-256x192.py) |  256x192   | 0.733 |      0.900      |      0.810      | 0.786 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-256x192-528ec248_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-256x192_20220926.log) |
| [pose_resnet_101_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-384x288.py) |  384x288   | 0.749 |      0.905      |      0.818      | 0.799 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-384x288-487d40a4_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_dark-8xb64-210e_coco-384x288_20220926.log) |
| [pose_resnet_152_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-256x192.py) |  256x192   | 0.743 |      0.906      |      0.819      | 0.796 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-256x192-f754df5f_20221031.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-256x192_20221031.log) |
| [pose_resnet_152_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-384x288.py) |  384x288   | 0.755 |      0.907      |      0.825      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-384x288-329f8454_20221031.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_dark-8xb32-210e_coco-384x288_20221031.log) |
