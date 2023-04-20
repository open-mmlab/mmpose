<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2204.12484">ViTPose (Neurips'2022)</a></summary>

```bibtex
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
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

The backbone models are pre-trained using MAE. The small-size pre-trained backbone can be found in [link](https://github.com/ViTAE-Transformer/ViTPose). The base, large, and huge pre-trained backbones can be found in [link](https://github.com/facebookresearch/mae).

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                                                                                             | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |    ckpt    |    log    |
| :--------------------------------------------------------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :--------: | :-------: |
| [ViTPose-S](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_small_coco_256x192.py)               |  256x192   | 0.738 |      0.903      |      0.813      | 0.792 |      0.940      | [ckpt](<>) | [log](<>) |
| [ViTPose-B](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_base_coco_256x192.py)                |  256x192   | 0.758 |      0.907      |      0.832      | 0.811 |      0.946      | [ckpt](<>) | [log](<>) |
| [ViTPose-L](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_large_coco_256x192.py)               |  256x192   | 0.783 |      0.914      |      0.852      | 0.835 |      0.953      | [ckpt](<>) | [log](<>) |
| [ViTPose-H](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_huge_coco_256x192.py)                |  256x192   | 0.791 |      0.917      |      0.857      | 0.841 |      0.954      | [ckpt](<>) | [log](<>) |
| [ViTPose-Simple-S](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_simple_small_coco_256x192.py) |  256x192   | 0.735 |      0.900      |      0.811      | 0.789 |      0.940      | [ckpt](<>) | [log](<>) |
| [ViTPose-Simple-B](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_simple_base_coco_256x192.py)  |  256x192   | 0.755 |      0.906      |      0.829      | 0.809 |      0.946      | [ckpt](<>) | [log](<>) |
| [ViTPose-Simple-L](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_simple_large_coco_256x192.py) |  256x192   | 0.782 |      0.914      |      0.853      | 0.834 |      0.953      | [ckpt](<>) | [log](<>) |
| [ViTPose-Simple-H](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitpose_simple_huge_coco_256x192.py)  |  256x192   | 0.789 |      0.916      |      0.856      | 0.840 |      0.954      | [ckpt](<>) | [log](<>) |
