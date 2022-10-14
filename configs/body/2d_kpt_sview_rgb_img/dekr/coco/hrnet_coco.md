<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.02300">DEKR (CVPR'2021)</a></summary>

```bibtex
@inproceedings{geng2021bottom,
  title={Bottom-up human pose estimation via disentangled keypoint regression},
  author={Geng, Zigang and Sun, Ke and Xiao, Bin and Zhang, Zhaoxiang and Wang, Jingdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14676--14686},
  year={2021}
}
```

</details>

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

Results on COCO val2017 without multi-scale test

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [HRNet-w32](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w32_coco_512x512.py) |  512x512   | 0.680 |      0.868      |      0.745      | 0.728 |      0.897      | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/dekr/hrnet_w32_coco_512x512-2a3056de_20220928.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/dekr/hrnet_w32_coco_512x512-20220928.log.json) |
| [HRNet-w48](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w48_coco_640x640.py) |  640x640   | 0.709 |      0.876      |      0.773      | 0.758 |      0.909      | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/dekr/hrnet_w48_coco_640x640-8854b2f1_20220930.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/dekr/hrnet_w48_coco_640x640-20220930.log.json) |

Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch                                                                | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                                 ckpt                                 |
| :------------------------------------------------------------------ | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :------------------------------------------------------------------: |
| [HRNet-w32](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w32_coco_512x512_multiscale.py)\* |  512x512   | 0.705 |      0.878      |      0.767      | 0.759 |      0.921      | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/dekr/hrnet_w32_coco_512x512-2a3056de_20220928.pth) |
| [HRNet-w48](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w48_coco_640x640_multiscale.py)\* |  640x640   | 0.722 |      0.882      |      0.785      | 0.778 |      0.928      | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/dekr/hrnet_w48_coco_640x640-8854b2f1_20220930.pth) |

\* these configs are generally used for evaluation. The training settings are identical to their single-scale counterparts.

The results of models provided by the authors on COCO val2017 using the same evaluation protocol

| Arch      | Input Size |   Setting    |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                             ckpt                             |
| :-------- | :--------: | :----------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :----------------------------------------------------------: |
| HRNet-w32 |  512x512   | single-scale | 0.678 |      0.868      |      0.744      | 0.728 |      0.897      | see [official implementation](https://github.com/HRNet/DEKR) |
| HRNet-w48 |  640x640   | single-scale | 0.707 |      0.876      |      0.773      | 0.757 |      0.909      | see [official implementation](https://github.com/HRNet/DEKR) |
| HRNet-w32 |  512x512   | multi-scale  | 0.708 |      0.880      |      0.773      | 0.763 |      0.921      | see [official implementation](https://github.com/HRNet/DEKR) |
| HRNet-w48 |  640x640   | multi-scale  | 0.721 |      0.881      |      0.786      | 0.779 |      0.927      | see [official implementation](https://github.com/HRNet/DEKR) |

The discrepancy between these results and that shown in paper is attributed to the differences in implementation details in evaluation process.
