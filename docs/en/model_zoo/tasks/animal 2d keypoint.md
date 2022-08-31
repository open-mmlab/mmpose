# Animal 2D Keypoint

<hr/>
<br/><br/>

## Locust Dataset

<br/>

### Animal 2d Keypoint + Resnet + Locust on Locust

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
| [pose_resnet_152](/configs/animal_2d_keypoint/topdown_heatmap/locust/td-hm_res152_8xb64-210e_locust-160x160.py) |  160x160   |  1.000  | 0.925 | 1.49 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160-4ea9b372_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160_20210407.log.json) |

<hr/>
<br/><br/>

## Animalpose Dataset

<br/>

### Animal 2d Keypoint + Resnet + Animalpose on Animalpose

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

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_res50_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.691 |      0.947      |      0.770      | 0.736 |      0.955      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256-e1f30bff_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_animalpose_256x256_20210426.log.json) |
| [pose_resnet_101](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_res101_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.696 |      0.948      |      0.774      | 0.736 |      0.951      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_animalpose_256x256-85563f4a_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_animalpose_256x256_20210426.log.json) |
| [pose_resnet_152](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_res152_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.704 |      0.938      |      0.786      | 0.748 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_animalpose_256x256-a0a7506c_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_animalpose_256x256_20210426.log.json) |

<br/>

### Animal 2d Keypoint + Hrnet + Animalpose on Animalpose

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

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

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.740 |      0.959      |      0.833      | 0.780 |      0.965      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256_20210426.log.json) |
| [pose_hrnet_w48](/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w48_8xb64-210e_animalpose-256x256.py) |  256x256   | 0.738 |      0.958      |      0.831      | 0.778 |      0.962      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256-34644726_20210426.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256_20210426.log.json) |

<hr/>
<br/><br/>

## Zebra Dataset

<br/>

### Animal 2d Keypoint + Resnet + Zebra on Zebra

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
<summary align="right"><a href="https://elifesciences.org/articles/47994">Grévy’s Zebra (Elife'2019)</a></summary>

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

Results on Grévy’s Zebra test set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_resnet_50](/configs/animal_2d_keypoint/topdown_heatmap/zebra/td-hm_res50_8xb64-210e_zebra-160x160.py) |  160x160   |  1.000  | 0.914 | 1.87 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_zebra_160x160-5a104833_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_zebra_160x160_20210407.log.json) |
| [pose_resnet_101](/configs/animal_2d_keypoint/topdown_heatmap/zebra/td-hm_res101_8xb64-210e_zebra-160x160.py) |  160x160   |  1.000  | 0.915 | 1.83 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_zebra_160x160-e8cb2010_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_zebra_160x160_20210407.log.json) |
| [pose_resnet_152](/configs/animal_2d_keypoint/topdown_heatmap/zebra/td-hm_res152_8xb64-210e_zebra-160x160.py) |  160x160   |  1.000  | 0.921 | 1.67 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_zebra_160x160-05de71dd_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_zebra_160x160_20210407.log.json) |

<hr/>
<br/><br/>

## Ap10k Dataset

<br/>

### Animal 2d Keypoint + Resnet + Ap10k on Ap10k

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

<br/>

### Animal 2d Keypoint + Hrnet + Ap10k on Ap10k

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

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
| [pose_hrnet_w32](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.722 |      0.935      |      0.789      |     0.557      |     0.729      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.log.json) |
| [pose_hrnet_w48](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.728 |      0.936      |      0.802      |     0.577      |     0.735      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_ap10k_256x256-d95ab412_20211029.log.json) |
