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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
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
| [pose_cspnext_t_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-tiny_udp_8xb256-210e_coco-256x192.py) |  256x192   | 0.665 |      0.874      |      0.723      | 0.723 |      0.917      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-coco_pt-in1k_210e-256x192-0908dd2d_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-coco_pt-in1k_210e-256x192-0908dd2d_20230123.json) |
| [pose_cspnext_s_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-s_udp_8xb256-210e_coco-256x192.py) |  256x192   | 0.697 |      0.886      |      0.776      | 0.753 |      0.929      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-coco_pt-in1k_210e-256x192-92dbfc1d_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-coco_pt-in1k_210e-256x192-92dbfc1d_20230123.json) |
| [pose_cspnext_m_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-m_udp_8xb256-210e_coco-256x192.py) |  256x192   | 0.732 |      0.896      |      0.806      | 0.785 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-coco_pt-in1k_210e-256x192-95f5967e_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-coco_pt-in1k_210e-256x192-95f5967e_20230123.json) |
| [pose_cspnext_l_udp](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-l_udp_8xb256-210e_coco-256x192.py) |  256x192   | 0.750 |      0.904      |      0.822      | 0.800 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-coco_pt-in1k_210e-256x192-661cdd8c_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-coco_pt-in1k_210e-256x192-661cdd8c_20230123.json) |
| [pose_cspnext_t_udp_aic_coco](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-tiny_udp_8xb256-210e_aic-coco-256x192.py) |  256x192   | 0.655 |      0.884      |      0.731      | 0.689 |      0.890      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.json) |
| [pose_cspnext_s_udp_aic_coco](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-s_udp_8xb256-210e_aic-coco-256x192.py) |  256x192   | 0.700 |      0.905      |      0.783      | 0.733 |      0.918      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.json) |
| [pose_cspnext_m_udp_aic_coco](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-m_udp_8xb256-210e_aic-coco-256x192.py) |  256x192   | 0.748 |      0.925      |      0.818      | 0.777 |      0.933      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.json) |
| [pose_cspnext_l_udp_aic_coco](/configs/body_2d_keypoint/topdown_heatmap/coco/cspnext-l_udp_8xb256-210e_aic-coco-256x192.py) |  256x192   | 0.772 |      0.936      |      0.839      | 0.799 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.json) |

Note that, UDP also adopts the unbiased encoding/decoding algorithm of [DARK](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/techniques.html#darkpose-cvpr-2020).

Flip test and detector is not used in the result of aic-coco training.
