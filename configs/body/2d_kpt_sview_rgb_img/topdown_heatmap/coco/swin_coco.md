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
<summary align="right"><a href="https://arxiv.org/abs/2103.14030">Swin (ICCV'2021)</a></summary>

```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

</details>

<!-- [OTHERS] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html">FPN (CVPR'2017)</a></summary>

```bibtex
@inproceedings{lin2017feature,
  title={Feature pyramid networks for object detection},
  author={Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2117--2125},
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
| [pose_swin_t](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_t_p4_w7_coco_256x192.py) |  256x192   | 0.724 |      0.901      |      0.806      | 0.782 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192-eaefe010_20220503.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192_20220503.log.json) |
| [pose_swin_b](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_b_p4_w7_coco_256x192.py) |  256x192   | 0.737 |      0.904      |      0.820      | 0.798 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192-7432be9e_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192_20220705.log.json) |
| [pose_swin_b](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_b_p4_w7_coco_384x288.py) |  384x288   | 0.759 |      0.910      |      0.832      | 0.811 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_384x288-3abf54f9_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_384x288_20220705.log.json) |
| [pose_swin_l](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_256x192.py) |  256x192   | 0.743 |      0.906      |      0.821      | 0.798 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192-642a89db_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192_20220705.log.json) |
| [pose_swin_l](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py) |  384x288   | 0.763 |      0.912      |      0.830      | 0.814 |      0.949      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288_20220705.log.json) |
| [pose_swin_b_fpn](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_b_p4_w7_fpn_coco_256x192.py) |  256x192   | 0.741 |      0.907      |      0.821      | 0.798 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_fpn_coco_256x192-a3b91c45_20220705.pth) | [log](https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_fpn_coco_256x192_20220705.log.json) |
