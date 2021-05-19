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
@article{labuguen2020macaquepose,
  title={MacaquePose: A novel ‘in the wild’macaque monkey pose dataset for markerless motion capture},
  author={Labuguen, Rollyn and Matsumoto, Jumpei and Negrete, Salvador and Nishimaru, Hiroshi and Nishijo, Hisao and Takada, Masahiko and Go, Yasuhiro and Inoue, Ken-ichi and Shibata, Tomohiro},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

#### Results on MacaquePose with ground-truth detection bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/macaque/res50_macaque_256x192.py)  | 256x192 | 0.799 | 0.952 | 0.919 | 0.837 | 0.964 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192-98f1dd3a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192_20210407.log.json) |
| [pose_resnet_101](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/macaque/res101_macaque_256x192.py) | 256x192 | 0.790 | 0.953 | 0.908 | 0.828 | 0.967 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_macaque_256x192-e3b9c6bb_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_macaque_256x192_20210407.log.json) |
| [pose_resnet_152](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/macaque/res152_macaque_256x192.py) | 256x192 | 0.794 | 0.951 | 0.915 | 0.834 | 0.968 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192-c42abc02_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192_20210407.log.json) |
