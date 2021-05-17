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

#### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/mobilenetv2_coco_512x512.py)  | 512x512 | 0.380 | 0.671 | 0.368 | 0.473 | 0.741 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512_20200816.log.json) |

#### Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/mobilenetv2_coco_512x512.py)  | 512x512 | 0.442 | 0.696 | 0.422 | 0.517 | 0.766 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512_20200816.log.json) |
