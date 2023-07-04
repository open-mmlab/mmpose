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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

<details>
<summary align="right"><a href="https://idea-research.github.io/HumanArt/">Human-Art (CVPR'2023)</a></summary>

```bibtex
@inproceedings{ju2023humanart,
    title={Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes},
    author={Ju, Xuan and Zeng, Ailing and Jianan, Wang and Qiang, Xu and Lei, Zhang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    year={2023}}
```

</details>

Results on Human-Art validation dataset with detector having human AP of 56.2 on Human-Art validation dataset

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) |  256x192   | 0.252 |      0.397      |      0.255      | 0.321 |      0.485      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_20220909.log) |
| [pose_hrnet_w32-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w32_8xb64-210e_humanart-256x192.py) |  256x192   | 0.399 |      0.545      |      0.420      | 0.466 |      0.613      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.json) |
| [pose_hrnet_w48-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py) |  256x192   | 0.271 |      0.413      |      0.277      | 0.339 |      0.499      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192_20220913.log) |
| [pose_hrnet_w48-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w48_8xb32-210e_humanart-256x192.py) |  256x192   | 0.417 |      0.553      |      0.442      | 0.481 |      0.617      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.json) |

Results on Human-Art validation dataset with ground-truth bounding-box

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) |  256x192   | 0.533 |      0.771      |      0.562      | 0.574 |      0.792      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_20220909.log) |
| [pose_hrnet_w32-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w32_8xb64-210e_humanart-256x192.py) |  256x192   | 0.754 |      0.906      |      0.812      | 0.783 |      0.916      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.json) |
| [pose_hrnet_w48-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py) |  256x192   | 0.557 |      0.782      |      0.593      | 0.595 |      0.804      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192_20220913.log) |
| [pose_hrnet_w48-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w48_8xb32-210e_humanart-256x192.py) |  256x192   | 0.769 |      0.906      |      0.825      | 0.796 |      0.919      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.json) |

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_hrnet_w32-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py) |  256x192   | 0.749 |      0.906      |      0.821      | 0.804 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_20220909.log) |
| [pose_hrnet_w32-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w32_8xb64-210e_humanart-256x192.py) |  256x192   | 0.741 |      0.902      |      0.814      | 0.795 |      0.941      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w32_8xb64-210e_humanart-256x192-0773ef0b_20230614.json) |
| [pose_hrnet_w48-coco](configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py) |  256x192   | 0.756 |      0.908      |      0.826      | 0.809 |      0.945      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192_20220913.log) |
| [pose_hrnet_w48-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_hrnet-w48_8xb32-210e_humanart-256x192.py) |  256x192   | 0.751 |      0.905      |      0.822      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_hrnet-w48_8xb32-210e_humanart-256x192-05178983_20230614.json) |
