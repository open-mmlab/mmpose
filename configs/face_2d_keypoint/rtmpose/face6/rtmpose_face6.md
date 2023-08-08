<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58580-8_27">RTMPose (arXiv'2023)</a></summary>

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.07399,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2212.07784">RTMDet (arXiv'2022)</a></summary>

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

- Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset.
- `Face6` and `*` denote model trained on 6 public datasets:
  - [COCO-Wholebody-Face](https://github.com/jin-s13/COCO-WholeBody/)
  - [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
  - [300W](https://ibug.doc.ic.ac.uk/resources/300-W/)
  - [COFW](http://www.vision.caltech.edu/xpburgos/ICCV13/)
  - [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody/)
  - [LaPa](https://github.com/JDAI-CV/lapa-dataset)

|                                    Config                                    | Input Size | NME<sup><br>(LaPa) | FLOPS<sup><br>(G) |                                    Download                                     |
| :--------------------------------------------------------------------------: | :--------: | :----------------: | :---------------: | :-----------------------------------------------------------------------------: |
| [RTMPose-t\*](./rtmpose/face_2d_keypoint/rtmpose-t_8xb256-120e_face6-256x256.py) |  256x256   |        1.67        |       0.652       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.pth) |
| [RTMPose-s\*](./rtmpose/face_2d_keypoint/rtmpose-s_8xb256-120e_face6-256x256.py) |  256x256   |        1.59        |       1.119       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-face6_pt-in1k_120e-256x256-d779fdef_20230529.pth) |
| [RTMPose-m\*](./rtmpose/face_2d_keypoint/rtmpose-m_8xb256-120e_face6-256x256.py) |  256x256   |        1.44        |       2.852       | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth) |
