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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [rtmpose_m](/configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.815  | 0.837 | 4.51 | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.json) |
