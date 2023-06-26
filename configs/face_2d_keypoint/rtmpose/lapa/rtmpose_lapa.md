<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2212.07784">RTMDet (ArXiv 2022)</a></summary>

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
<summary align="right"><a href="https://aaai.org/ojs/index.php/AAAI/article/view/6832/6686">LaPa (AAAI'2020)</a></summary>

```bibtex
@inproceedings{liu2020new,
  title={A New Dataset and Boundary-Attention Semantic Segmentation for Face Parsing.},
  author={Liu, Yinglu and Shi, Hailin and Shen, Hao and Si, Yue and Wang, Xiaobo and Mei, Tao},
  booktitle={AAAI},
  pages={11637--11644},
  year={2020}
}
```

</details>

Results on LaPa val set

| Arch                                                           | Input Size | NME  |                              ckpt                              |                              log                               |
| :------------------------------------------------------------- | :--------: | :--: | :------------------------------------------------------------: | :------------------------------------------------------------: |
| [pose_rtmpose_m](/configs/face_2d_keypoint/rtmpose/lapa/rtmpose-m_8xb64-120e_lapa-256x256.py) |  256x256   | 1.29 | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-lapa_pt-aic-coco_120e-256x256-762b1ae2_20230422.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-lapa_pt-aic-coco_120e-256x256-762b1ae2_20230422.json) |
