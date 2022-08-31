# Wholebody 2D Keypoint

<hr/>
<br/><br/>

## Coco-Wholebody Dataset

<br/>

### Wholebody 2d Keypoint + Resnet + Coco-Wholebody on Coco-Wholebody

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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [pose_resnet_50](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res50_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.652  |  0.738  |  0.615  |  0.749  |  0.606  |  0.715  |  0.46   |  0.584  |  0.521   |  0.633   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192-9e37ed88_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_50](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res50_8xb64-210e_coco-wholebody-384x288.py) |  384x288   |  0.666  |  0.747  |  0.634  |  0.763  |  0.731  |  0.811  |  0.536  |  0.646  |  0.574   |   0.67   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288-ce11e294_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_101](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res101_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.669  |  0.753  |  0.637  |  0.766  |  0.611  |  0.722  |  0.463  |  0.589  |  0.531   |  0.645   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192-7325f982_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_101](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res101_8xb64-210e_coco-wholebody-384x288.py) |  384x288   |  0.692  |  0.77   |  0.68   |  0.799  |  0.746  |  0.82   |  0.548  |  0.657  |  0.597   |  0.693   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288-6c137b9a_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_152](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res152_8xb32-210e_coco-wholebody-256x192.py) |  256x192   |  0.682  |  0.764  |  0.661  |  0.787  |  0.623  |  0.728  |  0.481  |  0.607  |  0.548   |  0.661   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192-5de8ae23_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_152](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res152_8xb32-210e_coco-wholebody-384x288.py) |  384x288   |  0.704  |  0.78   |  0.693  |  0.813  |  0.751  |  0.824  |  0.559  |  0.666  |   0.61   |  0.705   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288-eab8caa8_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288_20201004.log.json) |

<br/>

### Wholebody 2d Keypoint + Hrnet + Coco-Wholebody on Coco-Wholebody

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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [pose_hrnet_w32](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.678  |  0.755  |  0.543  |  0.661  |  0.63   |  0.708  |  0.467  |  0.566  |  0.536   |  0.636   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_20200918.log.json) |
| [pose_hrnet_w32](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-384x288.py) |  384x288   |   0.7   |  0.772  |  0.585  |  0.691  |  0.726  |  0.783  |  0.515  |  0.603  |  0.586   |  0.673   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_384x288-78cacac3_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_384x288_20200922.log.json) |
| [pose_hrnet_w48](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-256x192.py) |  256x192   |  0.701  |  0.776  |  0.675  |  0.787  |  0.656  |  0.743  |  0.535  |  0.639  |  0.579   |  0.681   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192_20200922.log.json) |
| [pose_hrnet_w48](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288.py) |  384x288   |  0.722  |  0.791  |  0.696  |  0.801  |  0.776  |  0.834  |  0.587  |  0.678  |  0.632   |  0.717   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_20200922.log.json) |

<br/>

### Wholebody 2d Keypoint + Hrnet + Dark + Coco-Wholebody on Coco-Wholebody

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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [pose_hrnet_w32_dark](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w32_dark-8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.693  |  0.764  |  0.564  |  0.674  |  0.737  |  0.809  |  0.503  |  0.602  |  0.582   |  0.671   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark_20200922.log.json) |
| [pose_hrnet_w48_dark+](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py) |  384x288   |  0.742  |  0.807  |  0.707  |  0.806  |  0.841  |  0.892  |  0.602  |  0.694  |  0.661   |  0.743   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark_20200918.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on COCO-WholeBody dataset. We find this will lead to better performance.

<br/>

### Wholebody 2d Keypoint + Vipnas + Coco-Wholebody on Coco-Wholebody

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2105.10154">ViPNAS (CVPR'2021)</a></summary>

```bibtex
@article{xu2021vipnas,
  title={ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search},
  author={Xu, Lumin and Guan, Yingda and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [S-ViPNAS-MobileNetV3](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-mbv3_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.619  |   0.7   |  0.477  |  0.608  |  0.585  |  0.689  |  0.386  |  0.505  |  0.473   |  0.578   | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192-0fee581a_20211205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_20211205.log.json) |
| [S-ViPNAS-Res50](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.643  |  0.726  |  0.553  |  0.694  |  0.587  |  0.698  |  0.41   |  0.529  |  0.495   |  0.607   | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192-49e1c3a4_20211112.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_20211112.log.json) |

<br/>

### Wholebody 2d Keypoint + Vipnas + Dark + Coco-Wholebody on Coco-Wholebody

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2105.10154">ViPNAS (CVPR'2021)</a></summary>

```bibtex
@article{xu2021vipnas,
  title={ViPNAS: Efficient Video Pose Estimation via Neural Architecture Search},
  author={Xu, Lumin and Guan, Yingda and Jin, Sheng and Liu, Wentao and Qian, Chen and Luo, Ping and Ouyang, Wanli and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
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

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                    | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR | Hand AP | Hand AR | Whole AP | Whole AR |                   ckpt                   |                   log                   |
| :-------------------------------------- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :------: | :--------------------------------------: | :-------------------------------------: |
| [S-ViPNAS-MobileNetV3_dark](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-mbv3_dark-8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.632  |  0.71   |  0.53   |  0.66   |  0.672  |  0.771  |  0.404  |  0.519  |  0.508   |  0.607   | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark-e2158108_20211205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark_20211205.log.json) |
| [S-ViPNAS-Res50_dark](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_dark-8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.65   |  0.732  |  0.55   |  0.686  |  0.684  |  0.783  |  0.437  |  0.554  |  0.528   |  0.632   | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark_20211112.log.json) |
