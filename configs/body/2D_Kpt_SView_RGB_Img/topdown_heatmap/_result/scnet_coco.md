<!-- [ALGORITHM] -->

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
  year={2020}
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

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_scnet_50](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/coco/scnet50_coco_256x192.py)   | 256x192 | 0.728 | 0.899 | 0.807 | 0.784 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192_20200709.log.json) |
| [pose_scnet_50](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/coco/scnet50_coco_384x288.py)   | 384x288 | 0.751 | 0.906 | 0.818 | 0.802 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288-9cacd0ea_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288_20200709.log.json) |
| [pose_scnet_101](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/coco/scnet101_coco_256x192.py)  | 256x192 | 0.733 | 0.903 | 0.813 | 0.790 | 0.941 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192-6d348ef9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192_20200709.log.json) |
| [pose_scnet_101](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/coco/scnet101_coco_384x288.py)  | 384x288 | 0.752 | 0.906 | 0.823 | 0.804 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288-0b6e631b_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288_20200709.log.json) |
