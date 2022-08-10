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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>

Results on Human3.6M test set with ground truth 2D detections

| Arch                                                         | Input Size | EPE  |  PCK  |                             ckpt                              |                             log                              |
| :----------------------------------------------------------- | :--------: | :--: | :---: | :-----------------------------------------------------------: | :----------------------------------------------------------: |
| [pose_hrnet_w32](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/h36m/hrnet_w32_h36m_256x256.py) |  256x256   | 9.43 | 0.911 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_h36m_256x256-d3206675_20210621.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_h36m_256x256_20210621.log.json) |
| [pose_hrnet_w48](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/h36m/hrnet_w48_h36m_256x256.py) |  256x256   | 7.36 | 0.932 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_h36m_256x256-78e88d08_20210621.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_h36m_256x256_20210621.log.json) |
