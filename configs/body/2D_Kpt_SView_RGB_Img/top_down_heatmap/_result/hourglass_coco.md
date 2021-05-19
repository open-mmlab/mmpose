<!-- [ALGORITHM] -->

```bibtex
@inproceedings{newell2016stacked,
  title={Stacked hourglass networks for human pose estimation},
  author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={483--499},
  year={2016},
  organization={Springer}
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
| [pose_hourglass_52](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/hourglass52_coco_256x256.py) | 256x256 | 0.726 | 0.896 | 0.799 | 0.780 | 0.934 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_256x256-4ec713ba_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_256x256_20200709.log.json) |
| [pose_hourglass_52](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/hourglass52_coco_384x384.py) | 384x384 | 0.746 | 0.900 | 0.813 | 0.797 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_384x384-be91ba2b_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_384x384_20200812.log.json) |
