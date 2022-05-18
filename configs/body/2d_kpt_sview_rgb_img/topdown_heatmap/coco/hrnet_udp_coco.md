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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
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
| [pose_hrnet_w32_udp](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp.py) |  256x192   | 0.760 |      0.907      |      0.827      | 0.811 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp-aba0be42_20210220.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_20210220.log.json) |
| [pose_hrnet_w32_udp](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288_udp.py) |  384x288   | 0.769 |      0.908      |      0.833      | 0.817 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp-e97c1a0f_20210223.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp_20210223.log.json) |
| [pose_hrnet_w48_udp](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_udp.py) |  256x192   | 0.767 |      0.906      |      0.834      | 0.817 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp-2554c524_20210223.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp_20210223.log.json) |
| [pose_hrnet_w48_udp](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_udp.py) |  384x288   | 0.772 |      0.910      |      0.835      | 0.820 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp-0f89c63e_20210223.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp_20210223.log.json) |
| [pose_hrnet_w32_udp_regress](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp_regress.py) |  256x192   | 0.758 |      0.908      |      0.823      | 0.812 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_regress-be2dbba4_20210222.pth) | [log](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_regress_20210222.log.json) |

Note that, UDP also adopts the unbiased encoding/decoding algorithm of [DARK](https://mmpose.readthedocs.io/en/latest/papers/techniques.html#div-align-center-darkpose-cvpr-2020-div).
