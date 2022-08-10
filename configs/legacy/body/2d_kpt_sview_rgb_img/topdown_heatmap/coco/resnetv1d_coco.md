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
| [pose_resnetv1d_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_256x192.py) |  256x192   | 0.722 |      0.897      |      0.799      | 0.777 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192-a243b840_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_384x288.py) |  384x288   | 0.730 |      0.900      |      0.799      | 0.780 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288-01f3fbb9_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288_20200727.log.json) |
| [pose_resnetv1d_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_256x192.py) |  256x192   | 0.731 |      0.899      |      0.809      | 0.786 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192-5bd08cab_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_384x288.py) |  384x288   | 0.748 |      0.902      |      0.816      | 0.799 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-5f9e421d_20200730.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-20200730.log.json) |
| [pose_resnetv1d_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_256x192.py) |  256x192   | 0.737 |      0.902      |      0.812      | 0.791 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192-c4df51dc_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_384x288.py) |  384x288   | 0.752 |      0.909      |      0.821      | 0.802 |      0.944      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-626c622d_20200730.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-20200730.log.json) |
