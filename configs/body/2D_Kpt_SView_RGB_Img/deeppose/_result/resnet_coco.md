<!-- [BACKBONE] -->

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
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
| [deeppose_resnet_50](/configs/body/2D_Kpt_SView_RGB_Img/deeppose/coco/deeppose_res50_coco_256x192.py)  | 256x192 | 0.526 | 0.816 | 0.586 | 0.638 | 0.887 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192_20210205.log.json) |
| [deeppose_resnet_101](/configs/body/2D_Kpt_SView_RGB_Img/deeppose/coco/deeppose_res101_coco_256x192.py) | 256x192 | 0.560 | 0.832 | 0.628 | 0.668 | 0.900 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192-2f247111_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_20210205.log.json) |
| [deeppose_resnet_152](/configs/body/2D_Kpt_SView_RGB_Img/deeppose/coco/deeppose_res152_coco_256x192.py) | 256x192 | 0.583 | 0.843 | 0.659 | 0.686 | 0.907 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_20210205.log.json) |
