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
<summary align="right"><a href="https://github.com/Fang-Haoshu/Halpe-FullBody/">AlphaPose (TPAMI'2022)</a></summary>

```bibtex
@article{alphapose,
  author = {Fang, Hao-Shu and Li, Jiefeng and Tang, Hongyang and Xu, Chao and Zhu, Haoyi and Xiu, Yuliang and Li, Yong-Lu and Lu, Cewu},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time},
  year = {2022}
}
```

</details>

- `*` denotes model trained on 7 public datasets:
  - [AI Challenger](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#aic)
  - [MS COCO](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#coco)
  - [CrowdPose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#crowdpose)
  - [MPII](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#mpii)
  - [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#sub-jhmdb-dataset)
  - [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe)
  - [PoseTrack18](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#posetrack18)
- `Body8` denotes the addition of the [OCHuman](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#ochuman) dataset, in addition to the 7 datasets mentioned above, for evaluation.

|                              Config                              | Input Size | PCK@0.1<sup><br>(Body8) | AUC<sup><br>(Body8) | Params(M) | FLOPS(G) |                              Download                               |
| :--------------------------------------------------------------: | :--------: | :---------------------: | :-----------------: | :-------: | :------: | :-----------------------------------------------------------------: |
| [RTMPose-t\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py) |  256x192   |          91.89          |        66.35        |   3.51    |   0.37   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth) |
| [RTMPose-s\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb1024-700e_body8-halpe26-256x192.py) |  256x192   |          93.01          |        68.62        |   5.70    |   0.70   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.pth) |
| [RTMPose-m\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py) |  256x192   |          94.75          |        71.91        |   13.93   |   1.95   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth) |
| [RTMPose-l\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb512-700e_body8-halpe26-256x192.py) |  256x192   |          95.37          |        73.19        |   28.11   |   4.19   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth) |
| [RTMPose-m\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py) |  384x288   |          95.15          |        73.56        |   14.06   |   4.37   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth) |
| [RTMPose-l\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb512-700e_body8-halpe26-384x288.py) |  384x288   |          95.56          |        74.38        |   28.24   |   9.40   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.pth) |
| [RTMPose-x\*](/configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py) |  384x288   |          95.74          |        74.82        |   50.00   |  17.29   | [Model](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth) |
