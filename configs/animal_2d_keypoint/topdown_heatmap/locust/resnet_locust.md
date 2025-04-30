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
<summary align="right"><a href="https://elifesciences.org/articles/47994">Desert Locust (Elife'2019)</a></summary>

```bibtex
@article{graving2019deepposekit,
  title={DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning},
  author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
  journal={Elife},
  volume={8},
  pages={e47994},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}
```

</details>

Results on Desert Locust test set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_resnet_50](/configs/animal_2d_keypoint/topdown_heatmap/locust/td-hm_res50_8xb64-210e_locust-160x160.py) |  160x160   |  1.000  | 0.900 | 2.27 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160-9efca22b_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160_20210407.log.json) |
| [pose_resnet_101](/configs/animal_2d_keypoint/topdown_heatmap/locust/td-hm_res101_8xb64-210e_locust-160x160.py) |  160x160   |  1.000  | 0.907 | 2.03 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160-d77986b3_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160_20210407.log.json) |
| [pose_resnet_152](/configs/animal_2d_keypoint/topdown_heatmap/locust/td-hm_res152_8xb32-210e_locust-160x160.py) |  160x160   |  1.000  | 0.925 | 1.49 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160-4ea9b372_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160_20210407.log.json) |
