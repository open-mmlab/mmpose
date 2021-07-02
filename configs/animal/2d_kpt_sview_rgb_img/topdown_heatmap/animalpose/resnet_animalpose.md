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
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Cao_Cross-Domain_Adaptation_for_Animal_Pose_Estimation_ICCV_2019_paper.html">Animal-Pose (ICCV'2019)</a></summary>

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

Results on AnimalPose validation set (1117 instances)

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/res50_animalpose_256x256.py)  | 256x256 | 0.688 | 0.945 | 0.772 | 0.733 | 0.952 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256_20210426.log.json) |
| [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/res101_animalpose_256x256.py) | 256x256 | 0.696 | 0.948 | 0.785 | 0.737 | 0.954 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_animalpose_256x256-85563f4a_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_animalpose_256x256_20210426.log.json) |
| [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/res152_animalpose_256x256.py) | 256x256 | 0.709 | 0.948 | 0.797 | 0.749 | 0.951 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_animalpose_256x256-a0a7506c_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_animalpose_256x256_20210426.log.json) |
