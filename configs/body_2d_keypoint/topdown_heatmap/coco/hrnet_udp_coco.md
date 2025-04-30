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
| [pose_hrnet_w32_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py) |  256x192   | 0.762 |      0.907      |      0.829      | 0.810 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192-73ede547_20220914.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192_20220914.log) |
| [pose_hrnet_w32_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288.py) |  384x288   | 0.768 |      0.909      |      0.832      | 0.815 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288-9a3f7c85_20220914.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_20220914.log) |
| [pose_hrnet_w48_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py) |  256x192   | 0.768 |      0.908      |      0.833      | 0.817 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192-3feaef8f_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192_20220913.log) |
| [pose_hrnet_w48_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py) |  384x288   | 0.773 |      0.911      |      0.836      | 0.821 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288-70d7ab01_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288_20220913.log) |
| [pose_hrnet_w32_udp_regress](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-regress-8xb64-210e_coco-256x192.py) |  256x192   | 0.759 |      0.907      |      0.827      | 0.813 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-regress-8xb64-210e_coco-256x192-9c0b77b4_20220926.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-regress-8xb64-210e_coco-256x192_20220226.log) |

Note that, UDP also adopts the unbiased encoding/decoding algorithm of [DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#darkpose-cvpr-2020).
