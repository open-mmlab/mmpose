<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html">ShufflenetV1 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{zhang2018shufflenet,
  title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6848--6856},
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
| [pose_shufflenetv1](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-256x192.py) |  256x192   | 0.587 |      0.849      |      0.654      | 0.654 |      0.896      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-256x192-7a7ea4f4_20221013.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-256x192_20221013.log) |
| [pose_shufflenetv1](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-384x288.py) |  384x288   | 0.626 |      0.862      |      0.696      | 0.687 |      0.903      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-384x288-8342f8ba_20221013.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_shufflenetv1_8xb64-210e_coco-384x288_20221013.log) |
