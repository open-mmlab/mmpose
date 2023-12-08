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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

- `Cocktail14` denotes model trained on 14 public datasets:
  - [AI Challenger](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#aic)
  - [CrowdPose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#crowdpose)
  - [MPII](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#mpii)
  - [sub-JHMDB](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#sub-jhmdb-dataset)
  - [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe)
  - [PoseTrack18](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#posetrack18)
  - [COCO-Wholebody](https://github.com/jin-s13/COCO-WholeBody/)
  - [UBody](https://github.com/IDEA-Research/OSX)
  - [Human-Art](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#human-art-dataset)
  - [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
  - [300W](https://ibug.doc.ic.ac.uk/resources/300-W/)
  - [COFW](http://www.vision.caltech.edu/xpburgos/ICCV13/)
  - [LaPa](https://github.com/JDAI-CV/lapa-dataset)
  - [InterHand](https://mks0601.github.io/InterHand2.6M/)

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                                      | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                            ckpt                            | log |
| :-------------------------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------------------------: | :-: |
| [rtmw-m](/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py) |  256x192   |  0.676  |  0.747  |  0.671  |  0.794  |  0.783  |  0.854  |  0.491  |  0.604  |  0.582   |  0.673   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth) |  -  |
| [rtmw-l](/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb1024-270e_cocktail14-256x192.py) |  256x192   |  0.743  |  0.807  |  0.763  |  0.868  |  0.834  |  0.889  |  0.598  |  0.701  |  0.660   |  0.746   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth) |  -  |
| [rtmw-x](/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb704-270e_cocktail14-256x192.py) |  256x192   |  0.746  |  0.808  |  0.770  |  0.869  |  0.844  |  0.896  |  0.610  |  0.710  |  0.672   |  0.752   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth) |  -  |
| [rtmw-l](/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py) |  384x288   |  0.761  |  0.824  |  0.793  |  0.885  |  0.884  |  0.921  |  0.663  |  0.752  |  0.701   |  0.780   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth) |  -  |
| [rtmw-x](/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py) |  384x288   |  0.763  |  0.826  |  0.796  |  0.888  |  0.884  |  0.923  |  0.664  |  0.755  |  0.702   |  0.781   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth) |  -  |
