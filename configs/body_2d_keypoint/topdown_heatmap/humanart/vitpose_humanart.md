To utilize ViTPose, you'll need to have [MMPreTrain](https://github.com/open-mmlab/mmpretrain). To install the required version, run the following command:

```shell
mim install 'mmpretrain>=1.0.0'
```

<!-- [BACKBONE] -->

<details>

<summary  align="right"><a  href="https://arxiv.org/abs/2204.12484">ViTPose (NeurIPS'2022)</a></summary>

```bibtex
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
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
| [ViTPose-S-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py) |  256x192   | 0.228 |      0.371      |      0.229      | 0.298 |      0.467      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.json) |
| [ViTPose-S-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-small_8xb64-210e_humanart-256x192.py) |  256x192   | 0.381 |      0.532      |      0.405      | 0.448 |      0.602      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.json) |
| [ViTPose-B-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py) |  256x192   | 0.270 |      0.423      |      0.272      | 0.340 |      0.510      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.json) |
| [ViTPose-B-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py) |  256x192   | 0.410 |      0.549      |      0.434      | 0.475 |      0.615      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.json) |

Results on Human-Art validation dataset with ground-truth bounding-box

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [ViTPose-S-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py) |  256x192   | 0.507 |      0.758      |      0.531      | 0.551 |      0.780      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.json) |
| [ViTPose-S-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-small_8xb64-210e_humanart-256x192.py) |  256x192   | 0.738 |      0.905      |      0.802      | 0.768 |      0.911      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.json) |
| [ViTPose-B-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py) |  256x192   | 0.555 |      0.782      |      0.590      | 0.599 |      0.809      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.json) |
| [ViTPose-B-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py) |  256x192   | 0.759 |      0.905      |      0.823      | 0.790 |      0.917      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.json) |

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

> With classic decoder

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [ViTPose-S-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py) |  256x192   | 0.739 |      0.903      |      0.816      | 0.792 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.json) |
| [ViTPose-S-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-small_8xb64-210e_humanart-256x192.py) |  256x192   | 0.737 |      0.902      |      0.811      | 0.792 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-small_8xb64-210e_humanart-256x192-5cbe2bfc_20230611.json) |
| [ViTPose-B-coco](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py) |  256x192   | 0.757 |      0.905      |      0.829      | 0.810 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.json) |
| [ViTPose-B-humanart-coco](configs/body_2d_keypoint/topdown_heatmap/humanart/td-hm_ViTPose-base_8xb64-210e_humanart-256x192.py) |  256x192   | 0.758 |      0.906      |      0.829      | 0.812 |      0.946      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/human_art/td-hm_ViTPose-base_8xb64-210e_humanart-256x192-b417f546_20230611.json) |
