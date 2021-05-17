<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
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
| [HRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/hrnet_w32_coco_512x512.py)  | 512x512 | 0.654 | 0.863 | 0.720 | 0.710 | 0.892 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_20200816.log.json) |
| [HRNet-w48](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/hrnet_w48_coco_512x512.py)  | 512x512 | 0.665 | 0.860 | 0.727 | 0.716 | 0.889 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_20200816.log.json) |

#### Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/hrnet_w32_coco_512x512.py)  | 512x512 | 0.698 | 0.877 | 0.760 | 0.748 | 0.907 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_20200816.log.json) |
| [HRNet-w48](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/hrnet_w48_coco_512x512.py)  | 512x512 | 0.712 | 0.880 | 0.771 | 0.757 | 0.909 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_20200816.log.json) |
