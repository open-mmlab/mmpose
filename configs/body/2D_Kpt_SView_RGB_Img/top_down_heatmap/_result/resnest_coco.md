<!-- [BACKBONE] -->

```bibtex
@article{zhang2020resnest,
  title={ResNeSt: Split-Attention Networks},
  author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2004.08955},
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

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnest_50](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/resnest50_coco_256x192.py)  | 256x192 | 0.721 | 0.899 | 0.802 | 0.776 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192-6e65eece_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192_20210320.log.json) |
| [pose_resnest_50](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/resnest50_coco_384x288.py)  | 384x288 | 0.737 | 0.900 | 0.811 | 0.789 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288-dcd20436_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288_20210320.log.json) |
| [pose_resnest_101](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/resnest101_coco_256x192.py) | 256x192 | 0.725 | 0.899 | 0.807 | 0.781 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192-2ffcdc9d_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192_20210320.log.json) |
| [pose_resnest_101](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/resnest101_coco_384x288.py) | 384x288 | 0.746 | 0.906 | 0.820 | 0.798 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288-80660658_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288_20210320.log.json) |
