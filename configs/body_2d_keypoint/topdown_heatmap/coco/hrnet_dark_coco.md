<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
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
| [pose_hrnet_w32_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192.py) |  256x192   | 0.757 |      0.907      |      0.825      | 0.807 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192-0e00bf12_20220914.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-256x192_20220914.log) |
| [pose_hrnet_w32_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-384x288.py) |  384x288   | 0.766 |      0.907      |      0.829      | 0.815 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-384x288-9bab4c9b_20220917.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_dark-8xb64-210e_coco-384x288_20220917.log) |
| [pose_hrnet_w48_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192.py) |  256x192   | 0.764 |      0.907      |      0.831      | 0.814 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192-e1ebdd6f_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-256x192_20220913.log) |
| [pose_hrnet_w48_dark](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py) |  384x288   | 0.772 |      0.911      |      0.833      | 0.821 |      0.948      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288_20220916.log) |
