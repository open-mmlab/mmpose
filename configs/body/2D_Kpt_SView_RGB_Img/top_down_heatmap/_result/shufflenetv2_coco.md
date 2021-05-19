<!-- [BACKBONE] -->

```bibtex
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
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
| [pose_shufflenetv2](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/shufflenetv2_coco_256x192.py)  | 256x192 | 0.599 | 0.854 | 0.663 | 0.664 | 0.899 | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192-0aba71c7_20200921.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192_20200921.log.json) |
| [pose_shufflenetv2](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/shufflenetv2_coco_384x288.py)  | 384x288 | 0.636 | 0.865 | 0.705 | 0.697 | 0.909 | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288-fb38ac3a_20200921.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288_20200921.log.json) |
