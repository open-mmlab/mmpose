<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58580-8_27">RSN (ECCV'2020)</a></summary>

```bibtex
@misc{cai2020learning,
    title={Learning Delicate Local Representations for Multi-Person Pose Estimation},
    author={Yuanhao Cai and Zhicheng Wang and Zhengxiong Luo and Binyi Yin and Angang Du and Haoqian Wang and Xinyu Zhou and Erjin Zhou and Xiangyu Zhang and Jian Sun},
    year={2020},
    eprint={2003.04030},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
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
| [rsn_18](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192.py) |  256x192   | 0.703 |      0.887      |      0.780      | 0.770 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192-72f4b4a7_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192_20201127.log.json) |
| [rsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.722 |      0.895      |      0.799      | 0.787 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192-72ffe709_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192_20201127.log.json) |
| [2xrsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xrsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.746 |      0.898      |      0.818      | 0.809 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192-50648f0e_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192_20201127.log.json) |
| [3xrsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_3xrsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.749 |      0.899      |      0.824      | 0.812 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192-58f57a68_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192_20201127.log.json) |
