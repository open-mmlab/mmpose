# Animal

<hr/>

## Animalpose

### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">Animal-Pose (ICCV'2019)</summary>

```bibtex
@InProceedings{Cao_2019_ICCV,
    author = {Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing},
    title = {Cross-Domain Adaptation for Animal Pose Estimation},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

</details>

#### Results on AnimalPose validation set (1117 instances)

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/res50_animalpose_256x256.py)  | 256x256 | 0.688 | 0.945 | 0.772 | 0.733 | 0.952 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256_20210426.log.json) |
| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/res101_animalpose_256x256.py) | 256x256 | 0.696 | 0.948 | 0.785 | 0.737 | 0.954 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_animalpose_256x256-85563f4a_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_animalpose_256x256_20210426.log.json) |
| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/res152_animalpose_256x256.py) | 256x256 | 0.709 | 0.948 | 0.797 | 0.749 | 0.951 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_animalpose_256x256-a0a7506c_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_animalpose_256x256_20210426.log.json) |


### Topdown_heatmap + Hrnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">HRNet (CVPR'2019)</summary>

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
<summary align="right">Animal-Pose (ICCV'2019)</summary>

```bibtex
@InProceedings{Cao_2019_ICCV,
    author = {Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing},
    title = {Cross-Domain Adaptation for Animal Pose Estimation},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

</details>

#### Results on AnimalPose validation set (1117 instances)

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w32_animalpose_256x256.py)  | 256x256 | 0.736 | 0.959 | 0.832 | 0.775 | 0.966 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256_20210426.log.json) |
| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w48_animalpose_256x256.py)  | 256x256 | 0.911 | 0.972 | 0.946 | 0.937 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256-34644726_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256_20210426.log.json) |


<hr/>

## Atrw

### Topdown_heatmap + Hrnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">HRNet (CVPR'2019)</summary>

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
<summary align="right">ATRW (ACM MM'2020)</summary>

```bibtex
@inproceedings{li2020atrw,
  title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
  author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2590--2598},
  year={2020}
}
```

</details>

#### Results on ATRW validation set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/hrnet_w32_atrw_256x256.py)  | 256x256 | 0.912 | 0.973 | 0.959 | 0.938 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_atrw_256x256-f027f09a_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_atrw_256x256_20210414.log.json) |
| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/hrnet_w48_atrw_256x256.py)  | 256x256 | 0.911 | 0.972 | 0.946 | 0.937 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_atrw_256x256-ac088892_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_atrw_256x256_20210414.log.json) |


### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">ATRW (ACM MM'2020)</summary>

```bibtex
@inproceedings{li2020atrw,
  title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
  author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2590--2598},
  year={2020}
}
```

</details>

#### Results on ATRW validation set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/res50_atrw_256x256.py)  | 256x256 | 0.900 | 0.973 | 0.932 | 0.929 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_atrw_256x256-546c4594_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_atrw_256x256_20210414.log.json) |
| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/res101_atrw_256x256.py) | 256x256 | 0.898 | 0.973 | 0.936 | 0.927 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_atrw_256x256-da93f371_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_atrw_256x256_20210414.log.json) |
| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/atrw/res152_atrw_256x256.py) | 256x256 | 0.896 | 0.973 | 0.931 | 0.927 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_atrw_256x256-2bb8e162_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_atrw_256x256_20210414.log.json) |


<hr/>

## Fly

### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">Vinegar Fly (Nature Methods'2019)</summary>

```bibtex
@article{pereira2019fast,
  title={Fast animal pose estimation using deep neural networks},
  author={Pereira, Talmo D and Aldarondo, Diego E and Willmore, Lindsay and Kislin, Mikhail and Wang, Samuel S-H and Murthy, Mala and Shaevitz, Joshua W},
  journal={Nature methods},
  volume={16},
  number={1},
  pages={117--125},
  year={2019},
  publisher={Nature Publishing Group}
}
```

</details>

#### Results on Vinegar Fly test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res50_fly_192x192.py) | 192x192 | 0.996 | 0.910 | 2.00 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_fly_192x192-5d0ee2d9_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_fly_192x192_20210407.log.json) |
|[pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res101_fly_192x192.py) | 192x192 | 0.996 | 0.912 | 1.95 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_fly_192x192-41a7a6cc_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_fly_192x192_20210407.log.json) |
|[pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res152_fly_192x192.py) | 192x192 | 0.997 | 0.917 | 1.78 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192-fcafbd5a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192_20210407.log.json) |


<hr/>

## Horse10

### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">Horse-10 (WACV'2021)</summary>

```bibtex
@inproceedings{mathis2021pretraining,
  title={Pretraining boosts out-of-domain robustness for pose estimation},
  author={Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1859--1868},
  year={2021}
}
```

</details>

#### Results on Horse-10 test set

|Set   | Arch  | Input Size | PCK@0.3 |  NME  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: |:------: |:------: |
|split1| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split1.py) | 256x256 | 0.956 | 0.113 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split2.py) | 256x256 | 0.954 | 0.111 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2-65e2a508_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split3.py) | 256x256 | 0.946 | 0.129 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3-9637d4eb_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res101_horse10_256x256-split1.py) | 256x256 | 0.958 | 0.115 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1-1b7c259c_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res101_horse10_256x256-split2.py) | 256x256 | 0.955 | 0.115 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2-30e2fa87_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res101_horse10_256x256-split3.py) | 256x256 | 0.946 | 0.126 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3-2eea5bb1_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res152_horse10_256x256-split1.py) | 256x256 | 0.969 | 0.105 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1-7e81fe2d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res152_horse10_256x256-split2.py) | 256x256 | 0.970 | 0.103 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2-3b3404a3_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res152_horse10_256x256-split3.py) | 256x256 | 0.957 | 0.131 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3-c957dac5_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3_20210405.log.json) |


### Topdown_heatmap + Hrnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">HRNet (CVPR'2019)</summary>

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
<summary align="right">Horse-10 (WACV'2021)</summary>

```bibtex
@inproceedings{mathis2021pretraining,
  title={Pretraining boosts out-of-domain robustness for pose estimation},
  author={Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1859--1868},
  year={2021}
}
```

</details>

#### Results on Horse-10 test set

|Set   | Arch  | Input Size | PCK@0.3 |  NME  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: |:------: |:------: |
|split1| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w32_horse10_256x256-split1.py) | 256x256 | 0.951 | 0.122 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split1-401d901a_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w32_horse10_256x256-split2.py) | 256x256 | 0.949 | 0.116 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split2-04840523_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w32_horse10_256x256-split3.py) | 256x256 | 0.939 | 0.153 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split3-4db47400_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w48_horse10_256x256-split1.py) | 256x256 | 0.973 | 0.095 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split1-3c950d3b_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w48_horse10_256x256-split2.py) | 256x256 | 0.969 | 0.101 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split2-8ef72b5d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w48_horse10_256x256-split3.py) | 256x256 | 0.961 | 0.128 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split3-0232ec47_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split3_20210405.log.json) |


<hr/>

## Locust

### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">Desert Locust (Elife'2019)</summary>

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

#### Results on Desert Locust test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/locust/res50_locust_160x160.py) | 160x160 | 0.999 | 0.899 | 2.27 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160-9efca22b_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160_20210407.log.json) |
|[pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/locust/res101_locust_160x160.py) | 160x160 | 0.999 | 0.907 | 2.03 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160-d77986b3_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160_20210407.log.json) |
|[pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/locust/res152_locust_160x160.py) | 160x160 | 1.000 | 0.926 | 1.48 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160-4ea9b372_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160_20210407.log.json) |


<hr/>

## Macaque

### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">MacaquePose (bioRxiv'2020)</summary>

```bibtex
@article{labuguen2020macaquepose,
  title={MacaquePose: A novel ‘in the wild’macaque monkey pose dataset for markerless motion capture},
  author={Labuguen, Rollyn and Matsumoto, Jumpei and Negrete, Salvador and Nishimaru, Hiroshi and Nishijo, Hisao and Takada, Masahiko and Go, Yasuhiro and Inoue, Ken-ichi and Shibata, Tomohiro},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

</details>

#### Results on MacaquePose with ground-truth detection bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res50_macaque_256x192.py)  | 256x192 | 0.799 | 0.952 | 0.919 | 0.837 | 0.964 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192-98f1dd3a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192_20210407.log.json) |
| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res101_macaque_256x192.py) | 256x192 | 0.790 | 0.953 | 0.908 | 0.828 | 0.967 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_macaque_256x192-e3b9c6bb_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_macaque_256x192_20210407.log.json) |
| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res152_macaque_256x192.py) | 256x192 | 0.794 | 0.951 | 0.915 | 0.834 | 0.968 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192-c42abc02_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192_20210407.log.json) |


### Topdown_heatmap + Hrnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">HRNet (CVPR'2019)</summary>

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
<summary align="right">MacaquePose (bioRxiv'2020)</summary>

```bibtex
@article{labuguen2020macaquepose,
  title={MacaquePose: A novel ‘in the wild’macaque monkey pose dataset for markerless motion capture},
  author={Labuguen, Rollyn and Matsumoto, Jumpei and Negrete, Salvador and Nishimaru, Hiroshi and Nishijo, Hisao and Takada, Masahiko and Go, Yasuhiro and Inoue, Ken-ichi and Shibata, Tomohiro},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```

</details>

#### Results on MacaquePose with ground-truth detection bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/hrnet_w32_macaque_256x192.py)  | 256x192 | 0.814 | 0.953 | 0.918 | 0.851 | 0.969 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192-f7e9e04f_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192_20210407.log.json) |
| [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/hrnet_w48_macaque_256x192.py)  | 256x192 | 0.818 | 0.963 | 0.917 | 0.855 | 0.971 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192_20210407.log.json) |


<hr/>

## Zebra

### Topdown_heatmap + Resnet

<!-- [ALGORITHM] -->

<details>
<summary align="right">SimpleBaseline2D (ECCV'2018)</summary>

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
<summary align="right">Desert Locust (Elife'2019)</summary>

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

#### Results on Grévy’s Zebra test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/zebra/res50_zebra_160x160.py) | 160x160 | 1.000 | 0.914 | 1.86 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_zebra_160x160-5a104833_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_zebra_160x160_20210407.log.json) |
|[pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/zebra/res101_zebra_160x160.py) | 160x160 | 1.000 | 0.916 | 1.82 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_zebra_160x160-e8cb2010_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_zebra_160x160_20210407.log.json) |
|[pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/zebra/res152_zebra_160x160.py) | 160x160 | 1.000 | 0.921 | 1.66 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_zebra_160x160-05de71dd_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_zebra_160x160_20210407.log.json) |
