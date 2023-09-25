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

- `Cocktail13` denotes model trained on 13 public datasets:
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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [rtmw-x](/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw-x_8xb320-270e_cocktail13-256x192.py) |  256x192   |  0.753  |  0.815  |  0.773  |  0.869  |  0.843  |  0.894  |  0.602  |  0.703  |  0.672   |  0.754   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail13_pt-ucoco_270e-256x192-fbef0d61_20230925.json) |
| [rtmw-x](/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw-x_8xb320-270e_cocktail13-384x288.py) |  384x288   |  0.764  |  0.825  |  0.791  |  0.883  |  0.882  |  0.922  |  0.654  |  0.744  |  0.702   |  0.779   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.json) |
