<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2312.07526">RTMO</a></summary>

```bibtex
@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

Results on COCO val2017

| Arch                                           | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) |                      ckpt                      |                      log                      |
| :--------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :----: | :----: | :----: | :--------------------------------------------: | :-------------------------------------------: |
| [RTMO-s](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-s_8xb32-700e_crowdpose-640x640.py) |  640x640   | 0.673 |      0.882      |      0.729      | 0.737  | 0.682  | 0.591  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-700e_crowdpose-640x640-79f81c0d_20231211.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-700e_crowdpose-640x640_20231211.json) |
| [RTMO-m](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-m_16xb16-700e_crowdpose-640x640.py) |  640x640   | 0.711 |      0.897      |      0.771      | 0.774  | 0.719  | 0.634  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rrtmo-m_16xb16-700e_crowdpose-640x640-0eaf670d_20231211.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-700e_crowdpose-640x640_20231211.json) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_crowdpose-640x640.py) |  640x640   | 0.732 |      0.907      |      0.793      | 0.792  | 0.741  | 0.653  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_crowdpose-640x640-1008211f_20231211.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_crowdpose-640x640_20231211.json) |
| [RTMO-l](/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py)\* |  640x640   | 0.838 |      0.947      |      0.893      | 0.888  | 0.847  | 0.772  | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_body7-crowdpose-640x640_20231219.json) |

\* indicates the model is trained using a combined dataset composed of AI Challenger, COCO, CrowdPose, Halpe, MPII, PoseTrack18 and sub-JHMDB.
