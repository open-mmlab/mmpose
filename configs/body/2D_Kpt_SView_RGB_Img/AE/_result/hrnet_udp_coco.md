<!-- [BACKBONE] -->

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

<!-- [ALGORITHM] -->

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
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
| [HRNet-w32_udp](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/hrnet_w32_coco_512x512_udp.py)  | 512x512 | 0.671 | 0.863 | 0.729 | 0.717 | 0.889 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp-91663bf9_20210220.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp_20210220.log.json) |
| [HRNet-w48_udp](/configs/body/2D_Kpt_SView_RGB_Img/AE/coco/hrnet_w48_coco_512x512_udp.py)  | 512x512 | 0.681 | 0.872 | 0.741 | 0.725 | 0.892 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_udp-de08fd8c_20210222.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_udp_20210222.log.json) |
