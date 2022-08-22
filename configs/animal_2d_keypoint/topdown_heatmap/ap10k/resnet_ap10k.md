<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html">SimpleBaseline2D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2108.12617">AP-10K (NeurIPS'2021)</a></summary>

```bibtex
@misc{yu2021ap10k,
      title={AP-10K: A Benchmark for Animal Pose Estimation in the Wild},
      author={Hang Yu and Yufei Xu and Jing Zhang and Wei Zhao and Ziyu Guan and Dacheng Tao},
      year={2021},
      eprint={2108.12617},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

Results on AP-10K validation set

| Arch                                       | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> |                    ckpt                     |                    log                     |
| :----------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :------------: | :------------: | :-----------------------------------------: | :----------------------------------------: |
| [pose_resnet_50](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_res50_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.680 |      0.926      |      0.738      |     0.552      |     0.687      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_ap10k_256x256-35760eb8_20211029.log.json) |
| [pose_resnet_101](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_res101_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.681 |      0.921      |      0.751      |     0.545      |     0.690      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_ap10k_256x256-9edfafb9_20211029.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_ap10k_256x256-9edfafb9_20211029.log.json) |
