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

<details>
<summary align="right"><a href="https://idea-research.github.io/HumanArt/">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023humanart,
    title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
    author={Ju, Xuan and Zeng, Ailing and Jianan, Wang and Qiang, Xu and Lei, Zhang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    year={2023}}
```

</details>

Results on Human-Art validation dataset with detector having human AP of 56.2 on Human-Art validation dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [rtmpose-t-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py) |  256x192   | 0.161 |      0.283      |      0.154      | 0.221 |      0.373      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) |
| [rtmpose-t-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-t_8xb256-420e_humanart-256x192.py) |  256x192   | 0.249 |      0.395      |      0.256      | 0.323 |      0.485      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.json) |
| [rtmpose-s-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.199 |      0.328      |      0.198      | 0.261 |      0.418      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) |
| [rtmpose-s-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-s_8xb256-420e_humanart-256x192.py) |  256x192   | 0.311 |      0.462      |      0.323      | 0.381 |      0.540      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.json) |
| [rtmpose-m-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   | 0.239 |      0.372      |      0.243      | 0.302 |      0.455      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) |
| [rtmpose-m-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-m_8xb256-420e_humanart-256x192.py) |  256x192   | 0.355 |      0.503      |      0.377      | 0.417 |      0.568      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.json) |
| [rtmpose-l-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   | 0.260 |      0.393      |      0.267      | 0.323 |      0.472      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) |
| [rtmpose-l-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-l_8xb256-420e_humanart-256x192.py) |  256x192   | 0.378 |      0.521      |      0.399      | 0.442 |      0.584      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.json) |

Results on Human-Art validation dataset with ground-truth bounding-box

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [rtmpose-t-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py) |  256x192   | 0.444 |      0.725      |      0.453      | 0.488 |      0.750      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) |
| [rtmpose-t-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-t_8xb256-420e_humanart-256x192.py) |  256x192   | 0.655 |      0.872      |      0.720      | 0.693 |      0.890      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.json) |
| [rtmpose-s-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.480 |      0.739      |      0.498      | 0.521 |      0.763      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) |
| [rtmpose-s-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-s_8xb256-420e_humanart-256x192.py) |  256x192   | 0.698 |      0.893      |      0.768      | 0.732 |      0.903      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.json) |
| [rtmpose-m-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   | 0.532 |      0.765      |      0.563      | 0.571 |      0.789      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) |
| [rtmpose-m-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-m_8xb256-420e_humanart-256x192.py) |  256x192   | 0.728 |      0.895      |      0.791      | 0.759 |      0.906      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.json) |
| [rtmpose-l-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   | 0.564 |      0.789      |      0.602      | 0.599 |      0.808      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) |
| [rtmpose-l-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-l_8xb256-420e_humanart-256x192.py) |  256x192   | 0.753 |      0.905      |      0.812      | 0.783 |      0.915      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.json) |

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [rtmpose-t-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py) |  256x192   | 0.682 |      0.883      |      0.759      | 0.736 |      0.920      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.json) |
| [rtmpose-t-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-t_8xb256-420e_humanart-256x192.py) |  256x192   | 0.665 |      0.875      |      0.739      | 0.721 |      0.916      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.json) |
| [rtmpose-s-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py) |  256x192   | 0.716 |      0.892      |      0.789      | 0.768 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.json) |
| [rtmpose-s-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-s_8xb256-420e_humanart-256x192.py) |  256x192   | 0.706 |      0.888      |      0.780      | 0.759 |      0.928      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.json) |
| [rtmpose-m-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py) |  256x192   | 0.746 |      0.899      |      0.817      | 0.795 |      0.935      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.json) |
| [rtmpose-m-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-m_8xb256-420e_humanart-256x192.py) |  256x192   | 0.725 |      0.892      |      0.795      | 0.775 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.json) |
| [rtmpose-l-coco](/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py) |  256x192   | 0.758 |      0.906      |      0.826      | 0.806 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.json) |
| [rtmpose-l-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-l_8xb256-420e_humanart-256x192.py) |  256x192   | 0.748 |      0.901      |      0.816      | 0.796 |      0.938      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.json) |

Results on COCO val2017 with ground-truth bounding box

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [rtmpose-t-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-t_8xb256-420e_humanart-256x192.py) |  256x192   | 0.679 |      0.895      |      0.755      | 0.710 |      0.907      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_8xb256-420e_humanart-256x192-60b68c98_20230612.json) |
| [rtmpose-s-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-s_8xb256-420e_humanart-256x192.py) |  256x192   | 0.725 |      0.916      |      0.798      | 0.753 |      0.925      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_8xb256-420e_humanart-256x192-5a3ac943_20230611.json) |
| [rtmpose-m-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-m_8xb256-420e_humanart-256x192.py) |  256x192   | 0.744 |      0.916      |      0.818      | 0.770 |      0.930      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_8xb256-420e_humanart-256x192-8430627b_20230611.json) |
| [rtmpose-l-humanart-coco](/configs/body_2d_keypoint/rtmpose/humanart/rtmpose-l_8xb256-420e_humanart-256x192.py) |  256x192   | 0.770 |      0.927      |      0.840      | 0.794 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_8xb256-420e_humanart-256x192-389f2cb0_20230611.json) |
