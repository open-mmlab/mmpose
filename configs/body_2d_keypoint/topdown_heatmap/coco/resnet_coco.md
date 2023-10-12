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
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py) |  256x192   | 0.718 |      0.898      |      0.796      | 0.774 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192-04af38ce_20220923.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192_20220923.log) |
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-384x288.py) |  384x288   | 0.731 |      0.900      |      0.799      | 0.782 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-384x288-7b8db90e_20220923.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-384x288_20220923.log) |
| [pose_resnet_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192.py) |  256x192   | 0.728 |      0.904      |      0.809      | 0.783 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192-065d3625_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192_20220926.log) |
| [pose_resnet_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb32-210e_coco-384x288.py) |  384x288   | 0.749 |      0.906      |      0.817      | 0.799 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192-065d3625_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res101_8xb64-210e_coco-256x192_20220926.log) |
| [pose_resnet_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-256x192.py) |  256x192   | 0.736 |      0.904      |      0.818      | 0.791 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-256x192-0345f330_20220928.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-256x192_20220928.log) |
| [pose_resnet_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-384x288.py) |  384x288   | 0.750 |      0.908      |      0.821      | 0.800 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-384x288-7fbb906f_20220927.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res152_8xb32-210e_coco-384x288_20220927.log) |

The following model is equipped with a visibility prediction head and has been trained using COCO and AIC datasets.

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm-vis_res50_8xb64-210e_coco-aic-256x192-merge.py) |  256x192   | 0.729 |      0.900      |      0.807      | 0.783 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm-vis_res50_8xb64-210e_coco-aic-256x192-merge-21815b2c_20230726.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192_20220923.log) |
