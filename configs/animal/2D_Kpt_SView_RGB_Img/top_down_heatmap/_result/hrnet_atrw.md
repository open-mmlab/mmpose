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
@inproceedings{li2020atrw,
  title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
  author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2590--2598},
  year={2020}
}
```

#### Results on ATRW validation set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/atrw/hrnet_w32_atrw_256x256.py)  | 256x256 | 0.912 | 0.973 | 0.959 | 0.938 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_atrw_256x256-f027f09a_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_atrw_256x256_20210414.log.json) |
| [pose_hrnet_w48](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/atrw/hrnet_w48_atrw_256x256.py)  | 256x256 | 0.911 | 0.972 | 0.946 | 0.937 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_atrw_256x256-ac088892_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_atrw_256x256_20210414.log.json) |
