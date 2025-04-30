<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1801.07372v2">DSNT (2018)</a></summary>

```bibtex
@article{nibali2018numerical,
  title={Numerical Coordinate Regression with Convolutional Neural Networks},
  author={Nibali, Aiden and He, Zhen and Morgan, Stuart and Prendergast, Luke},
  journal={arXiv preprint arXiv:1801.07372},
  year={2018}
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
| [ipr_resnet_50_dsnt](/configs/body_2d_keypoint/integral_regression/coco/ipr_res50_dsnt-8xb64-210e_coco-256x256.py) |  256x256   | 0.674 |      0.870      |      0.744      | 0.764 |      0.928      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/integral_regression/coco/ipr_res50_dsnt-8xb64-210e_coco-256x256-441eedc0_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/integral_regression/coco/ipr_res50_dsnt-8xb64-210e_coco-256x256-441eedc0_20220913.log.json) |
