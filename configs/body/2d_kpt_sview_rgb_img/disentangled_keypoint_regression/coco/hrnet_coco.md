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

| Arch                                                                                                                 | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |    ckpt    |    log    |
| :------------------------------------------------------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------: | :-------: |
| [HRNet-w32](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w32_coco_512x512_dekr.py) |  512x512   | 0.680 |      0.867      |      0.748      | 0.729 |      0.896      | [ckpt](<>) | [log](<>) |
| [HRNet-w48](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w48_coco_640x640_dekr.py) |  640x640   | 0.707 |      0.875      |      0.771      | 0.759 |      0.910      | [ckpt](<>) | [log](<>) |

Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch                                                                                                                 | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |    ckpt    |    log    |
| :------------------------------------------------------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------: | :-------: |
| [HRNet-w32](/configs/body/2d_kpt_sview_rgb_img/disentangled_keypoint_regression/coco/hrnet_w32_coco_512x512_dekr_multiscale.py) |  512x512   | 0.704 |      0.875      |      0.769      | 0.756 |      0.914      | [ckpt](<>) | [log](<>) |
