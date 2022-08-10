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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2004.00945">Halpe (CVPR'2020)</a></summary>

```bibtex
@inproceedings{li2020pastanet,
  title={PaStaNet: Toward Human Activity Knowledge Engine},
  author={Li, Yong-Lu and Xu, Liang and Liu, Xinpeng and Huang, Xijie and Xu, Yue and Wang, Shiyi and Fang, Hao-Shu and Ma, Ze and Chen, Mingyang and Lu, Cewu},
  booktitle={CVPR},
  year={2020}
}
```

</details>

Results on Halpe v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                                       | Input Size | Whole AP | Whole AR |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :------: | :------: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_hrnet_w48_dark+](/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/halpe/hrnet_w48_halpe_384x288_dark_plus.py) |  384x288   |  0.527   |  0.620   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_halpe_384x288_dark_plus-d13c2588_20211021.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_halpe_384x288_dark_plus_20211021.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on Halpe dataset. We find this will lead to better performance.
