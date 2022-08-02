<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">Associative Embedding (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
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

Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py)  | 512x512 | 0.466 | 0.742 | 0.479 | 0.552 | 0.797 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512_20200816.log.json) |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_640x640.py)  | 640x640 | 0.479 | 0.757 | 0.487 | 0.566 | 0.810 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640-2046f9cb_20200822.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640_20200822.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res101_coco_512x512.py)  | 512x512 | 0.554 | 0.807 | 0.599 | 0.622 | 0.841 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512-e0c95157_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512_20200816.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res152_coco_512x512.py)  | 512x512 | 0.595 | 0.829 | 0.648 | 0.651 | 0.856 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512-364eb38d_20200822.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512_20200822.log.json) |

Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py)  | 512x512 | 0.503 | 0.765 | 0.521 | 0.591 | 0.821 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512_20200816.log.json) |
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_640x640.py)  | 640x640 | 0.525 | 0.784 | 0.542 | 0.610 | 0.832 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640-2046f9cb_20200822.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640_20200822.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res101_coco_512x512.py)  | 512x512 | 0.603 | 0.831 | 0.641 | 0.668 | 0.870 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512-e0c95157_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512_20200816.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res152_coco_512x512.py)  | 512x512 | 0.660 | 0.860 | 0.713 | 0.709 | 0.889 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512-364eb38d_20200822.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512_20200822.log.json) |
