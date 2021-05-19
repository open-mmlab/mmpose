<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
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
| [pose_hrnet_w32](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/macaque/hrnet_w32_macaque_256x192.py)  | 256x192 | 0.814 | 0.953 | 0.918 | 0.851 | 0.969 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192-f7e9e04f_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192_20210407.log.json) |
| [pose_hrnet_w48](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/macaque/hrnet_w48_macaque_256x192.py)  | 256x192 | 0.818 | 0.963 | 0.917 | 0.855 | 0.971 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192_20210407.log.json) |
