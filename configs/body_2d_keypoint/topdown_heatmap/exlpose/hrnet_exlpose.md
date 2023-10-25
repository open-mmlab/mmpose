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
<summary align="right"><a href="http://cg.postech.ac.kr/research/ExLPose/">ExLPose (2023)</a></summary>

```bibtex
@inproceedings{ExLPose_2023_CVPR,
 title={Human Pose Estimation in Extremely Low-Light Conditions},
 author={Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim, ByungJu Woo, Haechan Lee, Sunghyun Cho, Suha Kwak},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2023}
}
```

</details>

Results on ExLPose-LLA val set with ground-truth bounding boxes

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-210e_exlpose-256x192.py) |  256x192   | 0.401 |      0.64       |      0.40       | 0.452 |      0.693      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-210e_exlpose-ll-256x192.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-210e_exlpose-ll-256x192.json) |
