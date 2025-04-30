<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.03332">SimCC (ECCV'2022)</a></summary>

```bibtex
@misc{https://doi.org/10.48550/arxiv.2107.03332,
  title={SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation},
  author={Li, Yanjie and Yang, Sen and Liu, Peidong and Zhang, Shoukui and Wang, Yunxiao and Wang, Zhicheng and Yang, Wankou and Xia, Shu-Tao},
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
| [simcc_resnet_50](/configs/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192.py) |  256x192   | 0.721 |      0.897      |      0.798      | 0.781 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.log.json) |
| [simcc_resnet_50](/configs/body_2d_keypoint/simcc/coco/simcc_res50_8xb32-140e_coco-384x288.py) |  384x288   | 0.735 |      0.899      |      0.800      | 0.790 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb32-140e_coco-384x288-45c3ba34_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb32-140e_coco-384x288-45c3ba34_20220913.log.json) |
