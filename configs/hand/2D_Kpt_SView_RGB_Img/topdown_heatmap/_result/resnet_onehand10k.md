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
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

#### Results on OneHand10K val set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py) | 256x256 | 0.989 | 0.555 | 25.19 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256-739c8639_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256_20210330.log.json) |
