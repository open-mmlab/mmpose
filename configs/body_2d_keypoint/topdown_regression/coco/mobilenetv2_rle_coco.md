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
| [deeppose_mobilenetv2_rle_pretrained](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_mobilenetv2_rle-pretrained-8xb64-210e_coco-256x192.py) |  256x192   | 0.593 |      0.836      |      0.660      | 0.644 |      0.877      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_regression/coco/td-reg_mobilenetv2_rle-pretrained-8xb64-210e_coco-256x192-39b73bd5_20220922.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_regression/coco/td-reg_mobilenetv2_rle-pretrained-8xb64-210e_coco-256x192-39b73bd5_20220922.log.json) |
