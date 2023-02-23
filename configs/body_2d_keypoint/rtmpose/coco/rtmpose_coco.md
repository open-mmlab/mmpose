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
| [rtmpose-t](./rtmpose-tiny_8xb256-420e_coco-256x192.py) |  256x192   | 0.682 |      0.883      |      0.759      | 0.736 |      0.920      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) |
| [rtmpose-s](./rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.716 |      0.892      |      0.789      | 0.768 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) |
| [rtmpose-m](./rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   | 0.746 |      0.899      |      0.817      | 0.795 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) |
| [rtmpose-l](./rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   | 0.758 |      0.906      |      0.826      | 0.806 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) |
| [rtmpose-t-aic-coco](./rtmpose-tiny_8xb256-420e_aic-coco-256x192.py) |  256x192   | 0.685 |      0.880      |      0.761      | 0.738 |      0.918      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.json) |
| [rtmpose-s-aic-coco](./rtmpose-s_8xb256-420e_aic-coco-256x192.py) |  256x192   | 0.722 |      0.892      |      0.794      | 0.772 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.json) |
| [rtmpose-m-aic-coco](./rtmpose-m_8xb256-420e_aic-coco-256x192.py) |  256x192   | 0.758 |      0.903      |      0.826      | 0.806 |      0.940      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.json) |
| [rtmpose-l-aic-coco](./rtmpose-l_8xb256-420e_aic-coco-256x192.py) |  256x192   | 0.765 |      0.906      |      0.835      | 0.813 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.json) |
