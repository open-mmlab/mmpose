<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html">ResNext (CVPR'2017)</a></summary>

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
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
| [pose_resnext_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext50_8xb64-210e_coco-256x192.py) |  256x192   | 0.715 |      0.897      |      0.791      | 0.771 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192-dcff15f6_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192_20200727.log.json) |
| [pose_resnext_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext50_8xb64-210e_coco-384x288.py) |  384x288   | 0.724 |      0.899      |      0.794      | 0.777 |      0.936      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288-412c848f_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288_20200727.log.json) |
| [pose_resnext_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext101_8xb64-210e_coco-256x192.py) |  256x192   | 0.726 |      0.900      |      0.801      | 0.781 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192-c7eba365_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192_20200727.log.json) |
| [pose_resnext_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext101_8xb32-210e_coco-384x288.py) |  384x288   | 0.744 |      0.903      |      0.815      | 0.794 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288-f5eabcd6_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288_20200727.log.json) |
| [pose_resnext_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext152_8xb32-210e_coco-256x192.py) |  256x192   | 0.730 |      0.903      |      0.808      | 0.785 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192-102449aa_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192_20200727.log.json) |
| [pose_resnext_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnext152_8xb48-210e_coco-384x288.py) |  384x288   | 0.742 |      0.904      |      0.810      | 0.794 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288-806176df_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288_20200727.log.json) |
