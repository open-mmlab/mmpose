<!-- [BACKBONE] -->

```bibtex
@inproceedings{zhang2018shufflenet,
  title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
  author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6848--6856},
  year={2018}
}
```

<!-- [DATASET] -->

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

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_shufflenetv1](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/coco/shufflenetv1_coco_256x192.py)  | 256x192 | 0.585 | 0.845 | 0.650 | 0.651 | 0.894 | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_256x192-353bc02c_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_256x192_20200727.log.json) |
| [pose_shufflenetv1](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/coco/shufflenetv1_coco_384x288.py)  | 384x288 | 0.622 | 0.859 | 0.685 | 0.684 | 0.901 | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_384x288-b2930b24_20200804.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_384x288_20200804.log.json) |
