<!-- [BACKBONE] -->

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
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
| [pose_mobilenetv2](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/mobilenetv2_coco_256x192.py)  | 256x192 | 0.646 | 0.874 | 0.723 | 0.707 | 0.917 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192_20200727.log.json) |
| [pose_mobilenetv2](/configs/body/2D_Kpt_SView_RGB_Img/top_down_heatmap/coco/mobilenetv2_coco_384x288.py)  | 384x288 | 0.673 | 0.879 | 0.743 | 0.729 | 0.916 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288_20200727.log.json) |
