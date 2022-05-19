<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.06403">LiteHRNet (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

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

</details>

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [LiteHRNet-18](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_18_coco_256x192.py) |  256x192   | 0.643 |      0.868      |      0.720      | 0.706 |      0.912      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192-6bace359_20211230.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192_20211230.log.json) |
| [LiteHRNet-18](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_18_coco_384x288.py) |  384x288   | 0.677 |      0.878      |      0.746      | 0.735 |      0.920      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_384x288-8d4dac48_20211230.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_384x288_20211230.log.json) |
| [LiteHRNet-30](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_256x192.py) |  256x192   | 0.675 |      0.881      |      0.754      | 0.736 |      0.924      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_256x192-4176555b_20210626.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_256x192_20210626.log.json) |
| [LiteHRNet-30](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_384x288.py) |  384x288   | 0.700 |      0.884      |      0.776      | 0.758 |      0.928      | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288-a3aef5c4_20210626.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288_20210626.log.json) |
