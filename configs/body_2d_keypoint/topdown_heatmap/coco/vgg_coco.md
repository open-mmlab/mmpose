<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1409.1556">VGG (ICLR'2015)</a></summary>

```bibtex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
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
| [vgg](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_vgg16-bn_8xb64-210e_coco-256x192.py) |  256x192   | 0.699 |      0.890      |      0.769      | 0.754 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/top_down/vgg/vgg16_bn_coco_256x192-7e7c58d6_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vgg/vgg16_bn_coco_256x192_20210517.log.json) |
