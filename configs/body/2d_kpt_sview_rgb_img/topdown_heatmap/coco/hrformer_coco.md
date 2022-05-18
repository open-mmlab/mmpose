<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html">HRFormer (NIPS'2021)</a></summary>

```bibtex
@article{yuan2021hrformer,
  title={HRFormer: High-Resolution Vision Transformer for Dense Predict},
  author={Yuan, Yuhui and Fu, Rao and Huang, Lang and Lin, Weihong and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
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
| [pose_hrformer_small](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192.py) |  256x192   | 0.738 |      0.904      |      0.811      | 0.792 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192-5310d898_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192_20220316.log.json) |
| [pose_hrformer_small](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_384x288.py) |  384x288   | 0.757 |      0.905      |      0.824      | 0.807 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_384x288-98d237ed_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_384x288_20220316.log.json) |
| [pose_hrformer_base](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_256x192.py) |  256x192   | 0.753 |      0.907      |      0.826      | 0.807 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192-6f5f1169_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192_20220316.log.json) |
| [pose_hrformer_base](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_384x288.py) |  384x288   | 0.774 |      0.909      |      0.842      | 0.823 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_384x288-ecf0758d_20220316.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192_20220316.log.json) |
