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
| [rsn_18](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192.py) |  256x192   | 0.704 |      0.887      |      0.781      | 0.773 |      0.927      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192-9049ed09_20221013.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn18_8xb32-210e_coco-256x192_20221013.log) |
| [rsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.724 |      0.894      |      0.799      | 0.790 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn50_8xb32-210e_coco-256x192-c35901d5_20221013.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_rsn50_8xb32-210e_coco-256x192_20221013.log) |
| [2xrsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xrsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.748 |      0.900      |      0.821      | 0.810 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xrsn50_8xb32-210e_coco-256x192-9ede341e_20221013.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_2xrsn50_8xb32-210e_coco-256x192_20221013.log) |
| [3xrsn_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_3xrsn50_8xb32-210e_coco-256x192.py) |  256x192   | 0.750 |      0.900      |      0.824      | 0.814 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_3xrsn50_8xb32-210e_coco-256x192-c3e3c4fe_20221013.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_3xrsn50_8xb32-210e_coco-256x192_20221013.log) |
