<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">Associative Embedding (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Lite_Pose_Efficient_Architecture_Design_for_2D_Human_Pose_Estimation_CVPR_2022_paper.html">LitePose (CVPR'2022)</a></summary>

```bibtex
@inproceedings{wang2022lite,
  title={Lite pose: Efficient architecture design for 2d human pose estimation},
  author={Wang, Yihan and Li, Muyang and Cai, Han and Chen, Wei-Ming and Han, Song},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13126--13136},
  year={2022}
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

Results on COCO val2017 without multi-scale test

| Arch                                                            | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                              ckpt                               |    log    |
| :-------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------------------------: | :-------: |
| [LitePose-XS](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/litepose_XS_coco_256x256.py) |  256x256   | 0.412 |      0.681      |      0.420      | 0.472 |      0.711      | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/litepose/litepose_xs_coco_256x256-02c29c82_20220714.pth) | [log](<>) |
| [LitePose-S](/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/litepose_S_coco_448x448.py) |  448x448   | 0.572 |      0.801      |      0.621      | 0.620 |      0.820      | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/litepose/litepose_s_coco_448x448-2bd10ac6_20220714.pth) | [log](<>) |
