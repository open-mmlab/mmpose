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
@InProceedings{Cao_2019_ICCV,
    author = {Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing},
    title = {Cross-Domain Adaptation for Animal Pose Estimation},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

#### Results on AnimalPose validation set (1117 instances)

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/animalpose/hrnet_w32_animalpose_256x256.py)  | 256x256 | 0.736 | 0.959 | 0.832 | 0.775 | 0.966 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256_20210426.log.json) |
| [pose_hrnet_w48](/configs/animal/2D_Kpt_SView_RGB_Img/top_down_heatmap/animalpose/hrnet_w48_animalpose_256x256.py)  | 256x256 | 0.911 | 0.972 | 0.946 | 0.937 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256-34644726_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256_20210426.log.json) |
