<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2102.12122">PVT (ICCV'2021)</a></summary>

```bibtex
@inproceedings{wang2021pyramid,
  title={Pyramid vision transformer: A versatile backbone for dense prediction without convolutions},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={568--578},
  year={2021}
}
```

</details>

<details>
<summary align="right"><a href="https://arxiv.org/abs/2106.13797">PVTV2 (CVMJ'2022)</a></summary>

```bibtex
@article{wang2022pvt,
  title={PVT v2: Improved baselines with Pyramid Vision Transformer},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={Computational Visual Media},
  pages={1--10},
  year={2022},
  publisher={Springer}
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
| [pose_pvt-s](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/pvt-s_coco_256x192.py) |  256x192   | 0.714 |      0.896      |      0.794      | 0.773 |      0.936      | [ckpt](https://download.openmmlab.com/mmpose/top_down/pvt/pvt_small_coco_256x192-4324a49d_20220501.pth) | [log](https://download.openmmlab.com/mmpose/top_down/pvt/pvt_small_coco_256x192_20220501.log.json) |
| [pose_pvtv2-b2](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/pvtv2-b2_coco_256x192.py) |  256x192   | 0.737 |      0.905      |      0.812      | 0.791 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/pvt/pvtv2_b2_coco_256x192-b4212737_20220501.pth) | [log](https://download.openmmlab.com/mmpose/top_down/pvt/pvtv2_b2_coco_256x192_20220501.log.json) |
