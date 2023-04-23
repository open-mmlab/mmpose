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
<summary align="right"><a href="https://arxiv.org/abs/2303.02760">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023human,
  title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
  author={Ju, Xuan and Zeng, Ailing and Wang, Jianan and Xu, Qiang and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

Results on Human-Art validation set with ground-truth bounding box

| Arch                                                            | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                              ckpt                               |    log    |
| :-------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------------------------: | :-------: |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/humanart/hrnet_w48_humanart_256x192.py) |  256x192   | 0.764 |      0.906      |      0.824      | 0.794 |      0.918      | [ckpt](https://drive.google.com/file/d/1gs1RCxRcItUHwA5N8P5_9mKcgwLiBjOO/view?usp=share_link) | [log](<>) |

Results on COCO val2017 set with ground-truth bounding box

| Arch                                                            | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                              ckpt                               |    log    |
| :-------------------------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------------------------: | :-------: |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/humanart/hrnet_w48_humanart_256x192.py) |  256x192   | 0.772 |      0.936      |      0.847      | 0.800 |      0.942      | [ckpt](https://drive.google.com/file/d/1gs1RCxRcItUHwA5N8P5_9mKcgwLiBjOO/view?usp=share_link) | [log](<>) |
