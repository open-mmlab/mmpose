<!-- [ALGORITHM] -->

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2212.07784">RTMDet (2022)</a></summary>

```bibtex
@misc{lyu2022rtmdet,
      title={RTMDet: An Empirical Study of Designing Real-Time Object Detectors},
      author={Chengqi Lyu and Wenwei Zhang and Haian Huang and Yue Zhou and Yudong Wang and Yanyi Liu and Shilong Zhang and Kai Chen},
      year={2022},
      eprint={2212.07784},
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
| [rtmpose-t](./rtmpose-tiny_8xb256-420e_coco-256x192.py) |  256x192   | 0.679 |      0.895      |      0.754      | 0.709 |      0.906      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) |
| [rtmpose-s](./rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.721 |      0.916      |      0.797      | 0.749 |      0.925      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) |
| [rtmpose-m](./rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   | 0.759 |      0.926      |      0.830      | 0.785 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) |
| [rtmpose-l](./rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   | 0.774 |      0.936      |      0.849      | 0.798 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) |
