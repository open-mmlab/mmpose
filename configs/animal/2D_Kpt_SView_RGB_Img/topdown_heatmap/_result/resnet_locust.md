<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

<!-- [DATASET] -->

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

#### Results on Desert Locust test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/2D_Kpt_SView_RGB_Img/topdown_heatmap/locust/res50_locust_160x160.py) | 160x160 | 0.999 | 0.899 | 2.27 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160-9efca22b_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160_20210407.log.json) |
|[pose_resnet_101](/configs/animal/2D_Kpt_SView_RGB_Img/topdown_heatmap/locust/res101_locust_160x160.py) | 160x160 | 0.999 | 0.907 | 2.03 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160-d77986b3_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160_20210407.log.json) |
|[pose_resnet_152](/configs/animal/2D_Kpt_SView_RGB_Img/topdown_heatmap/locust/res152_locust_160x160.py) | 160x160 | 1.000 | 0.926 | 1.48 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160-4ea9b372_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160_20210407.log.json) |
