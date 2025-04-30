<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html">ResNetV1D (CVPR'2019)</a></summary>

```bibtex
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
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

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnetv1d_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-256x192.py) |  256x192   | 0.722 |      0.897      |      0.796      | 0.777 |      0.936      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-256x192-27545d63_20221020.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-256x192_20221020.log) |
| [pose_resnetv1d_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-384x288.py) |  384x288   | 0.730 |      0.899      |      0.800      | 0.782 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-384x288-0646b46e_20221020.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d50_8xb64-210e_coco-384x288_20221020.log) |
| [pose_resnetv1d_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb64-210e_coco-256x192.py) |  256x192   | 0.732 |      0.901      |      0.808      | 0.785 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb64-210e_coco-256x192-ee9e7212_20221021.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb64-210e_coco-256x192_20221021.log) |
| [pose_resnetv1d_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb32-210e_coco-384x288.py) |  384x288   | 0.748 |      0.906      |      0.817      | 0.798 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb32-210e_coco-384x288-d0b5875f_20221028.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d101_8xb32-210e_coco-384x288_20221028.log) |
| [pose_resnetv1d_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb32-210e_coco-256x192.py) |  256x192   | 0.737 |      0.904      |      0.814      | 0.790 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb32-210e_coco-256x192-fd49f947_20221021.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb32-210e_coco-256x192_20221021.log) |
| [pose_resnetv1d_152](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb48-210e_coco-384x288.py) |  384x288   | 0.751 |      0.907      |      0.821      | 0.801 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb48-210e_coco-384x288-b9a99602_20221022.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnetv1d152_8xb48-210e_coco-384x288_20221022.log) |
