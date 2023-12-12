<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2204.06806">YOLO-Pose (CVPRW'2022)</a></summary>

```bibtex
@inproceedings{maji2022yolo,
  title={Yolo-pose: Enhancing yolo for multi person pose estimation using object keypoint similarity loss},
  author={Maji, Debapriya and Nagori, Soyeb and Mathew, Manu and Poddar, Deepak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
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
