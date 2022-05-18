<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.06403">LiteHRNet (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [LiteHRNet-18](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/litehrnet_18_mpii_256x256.py) |  256x256   | 0.859 |  0.260   | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_mpii_256x256-cabd7984_20210623.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_mpii_256x256_20210623.log.json) |
| [LiteHRNet-30](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/litehrnet_30_mpii_256x256.py) |  256x256   | 0.869 |  0.271   | [ckpt](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_mpii_256x256-faae8bd8_20210622.pth) | [log](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_mpii_256x256_20210622.log.json) |
