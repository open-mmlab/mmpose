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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html">MobilenetV2 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
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
| [simcc_mobilenetv2_wo_deconv](/configs/body_2d_keypoint/simcc/coco/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192.py) |  256x192   | 0.620 |      0.855      |      0.697      | 0.678 |      0.902      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192-4b0703bb_20221010.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192-4b0703bb_20221010.log.json) |
