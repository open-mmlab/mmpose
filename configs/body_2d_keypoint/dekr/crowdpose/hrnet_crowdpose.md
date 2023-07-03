<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.02300">DEKR (CVPR'2021)</a></summary>

```bibtex
@inproceedings{geng2021bottom,
  title={Bottom-up human pose estimation via disentangled keypoint regression},
  author={Geng, Zigang and Sun, Ke and Xiao, Bin and Zhang, Zhaoxiang and Wang, Jingdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14676--14686},
  year={2021}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
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

Results on CrowdPose test without multi-scale test

| Arch                                           | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) |                      ckpt                      |                      log                      |
| :--------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :----: | :----: | :----: | :--------------------------------------------: | :-------------------------------------------: |
| [HRNet-w32](/configs/body_2d_keypoint/dekr/crowdpose/dekr_hrnet-w32_8xb10-300e_crowdpose-512x512.py) |  512x512   | 0.663 |      0.857      |      0.714      | 0.740  | 0.671  | 0.576  | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/crowdpose/dekr_hrnet-w32_8xb10-300e_crowdpose-512x512_147bae97-20221228.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/crowdpose/dekr_hrnet-w32_8xb10-300e_crowdpose-512x512_20221228.json) |
| [HRNet-w48](/configs/body_2d_keypoint/dekr/crowdpose/dekr_hrnet-w48_8xb5-300e_crowdpose-640x640.py) |  640x640   | 0.679 |      0.869      |      0.731      | 0.753  | 0.688  | 0.593  | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/crowdpose/dekr_hrnet-w48_8xb5-300e_crowdpose-640x640_4ea6031e-20230128.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/crowdpose/dekr_hrnet-w48_8xb5-300e_crowdpose-640x640_20230128.json) |
