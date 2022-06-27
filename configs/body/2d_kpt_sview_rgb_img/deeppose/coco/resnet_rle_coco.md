<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.11291">RLE (ICCV'2021)</a></summary>

```bibtex
@inproceedings{li2021human,
  title={Human pose regression with residual log-likelihood estimation},
  author={Li, Jiefeng and Bian, Siyuan and Zeng, Ailing and Wang, Can and Pang, Bo and Liu, Wentao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11025--11034},
  year={2021}
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
| [deeppose_resnet_50_rle](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192_rle.py) |  256x192   | 0.704 |      0.883      |      0.777      | 0.751 |      0.920      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_rle-2ea9bb4a_20220616.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_rle_20220616.log.json) |
| [deeppose_resnet_101_rle](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res101_coco_256x192_rle.py) |  256x192   | 0.722 |      0.894      |      0.794      | 0.768 |      0.930      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_rle-16c3d461_20220615.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_rle_20220615.log.json) |
| [deeppose_resnet_152_rle](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res152_coco_256x192_rle.py) |  256x192   | 0.731 |      0.897      |      0.805      | 0.777 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_rle-c05bdccf_20220615.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_rle_20220615.log.json) |
| [deeppose_resnet_152_rle](/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res152_coco_384x288_rle.py) |  384x288   | 0.749 |      0.901      |      0.815      | 0.793 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle-b77c4c37_20220624.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_384x288_rle_20220624.log.json) |
