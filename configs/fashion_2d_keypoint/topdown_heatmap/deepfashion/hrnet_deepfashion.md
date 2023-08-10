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

<!-- [BACKBONE] -->

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html">DeepFashion (CVPR'2016)</a></summary>

```bibtex
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-46475-6_15">DeepFashion (ECCV'2016)</a></summary>

```bibtex
@inproceedings{liuYLWTeccv16FashionLandmark,
 author = {Liu, Ziwei and Yan, Sijie and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 title = {Fashion Landmark Detection in the Wild},
 booktitle = {European Conference on Computer Vision (ECCV)},
 month = {October},
 year = {2016}
 }
```

</details>

Results on DeepFashion val set

| Set   |                           Arch                            | Input Size | PCK@0.2 | AUC  | EPE  |                           ckpt                            |                           log                            |
| :---- | :-------------------------------------------------------: | :--------: | :-----: | :--: | :--: | :-------------------------------------------------------: | :------------------------------------------------------: |
| upper | [pose_hrnet_w48_udp](td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_uppder-256x192.py) |  256x192   |  96.1   | 60.9 | 15.1 | [ckpt](https://download.openmmlab.com/mmpose/v1/fashion_2d_keypoint/topdown_heatmap/deepfashion/td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_upper-256x192-de7c0eb1_20230810.pth) | [log](https://download.openmmlab.com/mmpose/v1/fashion_2d_keypoint/topdown_heatmap/deepfashion/td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_upper-256x192-de7c0eb1_20230810.log) |
| lower | [pose_hrnet_w48_udp](td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_lower-256x192.py) |  256x192   |  97.8   | 76.1 | 8.9  | [ckpt](https://download.openmmlab.com/mmpose/v1/fashion_2d_keypoint/topdown_heatmap/deepfashion/td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_lower-256x192-ddaf747d_20230810.pth) | [log](https://download.openmmlab.com/mmpose/v1/fashion_2d_keypoint/topdown_heatmap/deepfashion/td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_lower-256x192-ddaf747d_20230810.log) |
| full  | [pose_hrnet_w48_udp](td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_full-256x192.py) |  256x192   |  98.3   | 67.3 | 11.7 | [ckpt](https://download.openmmlab.com/mmpose/v1/fashion_2d_keypoint/topdown_heatmap/deepfashion/td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_full-256x192-7ab504c7_20230810.pth) | [log](https://download.openmmlab.com/mmpose/v1/fashion_2d_keypoint/topdown_heatmap/deepfashion/td-hm_hrnet-w48_udp_8xb32-210e_deepfashion_full-256x192-7ab504c7_20230810.log) |

Note: Due to the time constraints, we have only trained resnet50 models. We warmly welcome any contributions if you can successfully reproduce the results from the paper!
