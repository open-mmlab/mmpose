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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [rtmpose-m](./rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |  0.697  |  0.743  |  0.660  |  0.749  |  0.822  |  0.858  |  0.483  |  0.564  |  0.604   |  0.667   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.json) |
| [rtmpose-l](./rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |  0.721  |  0.764  |  0.693  |  0.780  |  0.844  |  0.876  |  0.523  |  0.600  |  0.632   |  0.694   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.json) |
| [rtmpose-l](./rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |  0.736  |  0.776  |  0.738  |  0.810  |  0.895  |  0.918  |  0.591  |  0.659  |  0.670   |  0.723   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.json) |
