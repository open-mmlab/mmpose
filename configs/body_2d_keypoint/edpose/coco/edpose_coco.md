<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/2302.01593.pdf">ED-Pose (ICLR'2023)</a></summary>

```bibtex
@inproceedings{
yang2023explicit,
title={Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation},
author={Jie Yang and Ailing Zeng and Shilong Liu and Feng Li and Ruimao Zhang and Lei Zhang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=s4WVupnJjmX}
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

Results on COCO val2017.

| Arch                                          | BackBone  |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                      ckpt                      |                      log                      |
| :-------------------------------------------- | :-------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------------------------------------------: | :-------------------------------------------: |
| [edpose_res50_coco](/configs/body_2d_keypoint/edpose/coco/edpose_res50_8xb2-50e_coco-800x1333.py) | ResNet-50 | 0.716 |      0.897      |      0.783      | 0.793 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/edpose/coco/edpose_res50_coco_3rdparty.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/edpose/coco/edpose_res50_coco_3rdparty.json) |

The checkpoint is converted from the official repo. The training of EDPose is not supported yet. It will be supported in the future updates.

The above config follows [Pure Python style](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta). Please install `mmengine>=0.8.2` to use this config.
