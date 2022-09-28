<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1901.00148">MSPN (ArXiv'2019)</a></summary>

```bibtex
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
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
| [mspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.723 |      0.895      |      0.794      | 0.788 |      0.934      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192-8fbfb5d0_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192_20201123.log.json) |
| [2xmspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xmspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.754 |      0.903      |      0.826      | 0.816 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192-c8765a5c_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192_20201123.log.json) |
| [3xmspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_3xmspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.758 |      0.904      |      0.830      | 0.821 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192-e348f18e_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192_20201123.log.json) |
| [4xmspn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_4xmspn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.765 |      0.906      |      0.835      | 0.826 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192_20201123.log.json) |
