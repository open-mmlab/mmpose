<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1711.08229">IPR (ECCV'2018)</a></summary>

```bibtex
@inproceedings{sun2018integral,
  title={Integral human pose regression},
  author={Sun, Xiao and Xiao, Bin and Wei, Fangyin and Liang, Shuang and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={529--545},
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
| [ipr_resnet_50](/configs/body_2d_keypoint/integral_regression/coco/ipr_res50_8xb64-210e_coco-256x256.py) |  256x256   | 0.633 |      0.860      |      0.703      | 0.730 |      0.919      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/integral_regression/coco/ipr_res50_8xb64-210e_coco-256x256-a3898a33_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/integral_regression/coco/ipr_res50_8xb64-210e_coco-256x256-a3898a33_20220913.log.json) |
