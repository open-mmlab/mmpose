# Face 2D Keypoint

<hr/>
<br/><br/>

## 300w Dataset

<br/>

### Face 2d Keypoint + Hrnetv2 + 300w on 300w

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.sciencedirect.com/science/article/pii/S0262885616000147">300W (IMAVIS'2016)</a></summary>

```bibtex
@article{sagonas2016300,
  title={300 faces in-the-wild challenge: Database and results},
  author={Sagonas, Christos and Antonakos, Epameinondas and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  journal={Image and vision computing},
  volume={47},
  pages={3--18},
  year={2016},
  publisher={Elsevier}
}
```

</details>

Results on 300W dataset

The model is trained on 300W train.

| Arch                               | Input Size | NME<sub>*common*</sub> | NME<sub>*challenge*</sub> | NME<sub>*full*</sub> | NME<sub>*test*</sub> |                ckpt                 |                log                 |
| :--------------------------------- | :--------: | :--------------------: | :-----------------------: | :------------------: | :------------------: | :---------------------------------: | :--------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/300w/td-hm_hrnetv2-w18_8xb64-60e_300w-256x256.py) |  256x256   |          2.92          |           5.64            |         3.45         |         4.10         | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256-eea53406_20211019.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256_20211019.log.json) |

<hr/>
<br/><br/>

## Coco_wholebody_face Dataset

<br/>

### Face 2d Keypoint + Hourglass + Coco + Wholebody + Face on Coco_wholebody_face

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29">Hourglass (ECCV'2016)</a></summary>

```bibtex
@inproceedings{newell2016stacked,
  title={Stacked hourglass networks for human pose estimation},
  author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={483--499},
  year={2016},
  organization={Springer}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_hourglass_52](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_hourglass52_8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0587 | [ckpt](https://download.openmmlab.com/mmpose/face/hourglass/hourglass52_coco_wholebody_face_256x256-6994cf2e_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/hourglass/hourglass52_coco_wholebody_face_256x256_20210909.log.json) |

<br/>

### Face 2d Keypoint + Resnet + Coco + Wholebody + Face on Coco_wholebody_face

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

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_res50](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_res50_8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0582 | [ckpt](https://download.openmmlab.com/mmpose/face/resnet/res50_coco_wholebody_face_256x256-5128edf5_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/resnet/res50_coco_wholebody_face_256x256_20210909.log.json) |

<br/>

### Face 2d Keypoint + Hrnetv2 + Dark + Coco + Wholebody + Face on Coco_wholebody_face

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_hrnetv2_w18_dark](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_hrnetv2-w18_dark-8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0513 | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_coco_wholebody_face_256x256_dark-3d9a334e_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_coco_wholebody_face_256x256_dark_20210909.log.json) |

<br/>

### Face 2d Keypoint + Mobilenetv2 + Coco + Wholebody + Face on Coco_wholebody_face

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html">MobilenetV2 (CVPR'2018)</a></summary>

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_mobilenetv2](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_mobilenetv2_8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0611 | [ckpt](https://download.openmmlab.com/mmpose/face/mobilenetv2/mobilenetv2_coco_wholebody_face_256x256-4a3f096e_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/mobilenetv2/mobilenetv2_coco_wholebody_face_256x256_20210909.log.json) |

<br/>

### Face 2d Keypoint + Hrnetv2 + Coco + Wholebody + Face on Coco_wholebody_face

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_hrnetv2-w18_8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0569 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_coco_wholebody_face_256x256-c1ca469b_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_coco_wholebody_face_256x256_20210909.log.json) |

<br/>

### Face 2d Keypoint + Scnet + Coco + Wholebody + Face on Coco_wholebody_face

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Improving_Convolutional_Networks_With_Self-Calibrated_Convolutions_CVPR_2020_paper.html">SCNet (CVPR'2020)</a></summary>

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Face (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Face val set

| Arch                                                          | Input Size |  NME   |                              ckpt                              |                              log                              |
| :------------------------------------------------------------ | :--------: | :----: | :------------------------------------------------------------: | :-----------------------------------------------------------: |
| [pose_scnet_50](/configs/face_2d_keypoint/topdown_heatmap/coco_wholebody_face/td-hm_scnet50_8xb32-60e_coco-wholebody-face-256x256.py) |  256x256   | 0.0567 | [ckpt](https://download.openmmlab.com/mmpose/face/scnet/scnet50_coco_wholebody_face_256x256-a0183f5f_20210909.pth) | [log](https://download.openmmlab.com/mmpose/face/scnet/scnet50_coco_wholebody_face_256x256_20210909.log.json) |

<hr/>
<br/><br/>

## WFLW Dataset

<br/>

### Face 2d Keypoint + Hrnetv2 + Dark + WFLW on WFLW

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
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
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Look_at_Boundary_CVPR_2018_paper.html">WFLW (CVPR'2018)</a></summary>

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

</details>

Results on WFLW dataset

The model is trained on WFLW train.

| Arch       | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |    ckpt    |    log    |
| :--------- | :--------: | :------------------: | :------------------: | :--------------------------: | :-----------------------: | :------------------: | :--------------------: | :------------------------: | :--------: | :-------: |
| [pose_hrnetv2_w18_dark](/configs/face_2d_keypoint/topdown_heatmap/wflw/td-hm_hrnetv2-w18_dark-8xb64-60e_wflw-256x256.py) |  256x256   |         3.98         |         6.98         |             3.96             |           4.78            |         4.56         |          3.89          |            4.29            | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark-3f8e0c2c_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark_20210125.log.json) |

<br/>

### Face 2d Keypoint + Hrnetv2 + Awing + WFLW on WFLW

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1904.07399.pdf">AdaptiveWingloss (ICCV'2019)</a></summary>

```bibtex
@inproceedings{wang2019adaptive,
  title={Adaptive wing loss for robust face alignment via heatmap regression},
  author={Wang, Xinyao and Bo, Liefeng and Fuxin, Li},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6971--6981},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Look_at_Boundary_CVPR_2018_paper.html">WFLW (CVPR'2018)</a></summary>

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

</details>

Results on WFLW dataset

The model is trained on WFLW train.

| Arch       | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |    ckpt    |    log    |
| :--------- | :--------: | :------------------: | :------------------: | :--------------------------: | :-----------------------: | :------------------: | :--------------------: | :------------------------: | :--------: | :-------: |
| [pose_hrnetv2_w18_awing](/configs/face_2d_keypoint/topdown_heatmap/wflw/td-hm_hrnetv2-w18_awing-8xb64-60e_wflw-256x256.py) |  256x256   |         4.02         |         6.94         |             3.97             |           4.78            |         4.59         |          3.87          |            4.28            | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_awing-5af5055c_20211212.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_awing_20211212.log.json) |

<br/>

### Face 2d Keypoint + Hrnetv2 + WFLW on WFLW

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Look_at_Boundary_CVPR_2018_paper.html">WFLW (CVPR'2018)</a></summary>

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

</details>

Results on WFLW dataset

The model is trained on WFLW train.

| Arch       | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |    ckpt    |    log    |
| :--------- | :--------: | :------------------: | :------------------: | :--------------------------: | :-----------------------: | :------------------: | :--------------------: | :------------------------: | :--------: | :-------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/wflw/td-hm_hrnetv2-w18_8xb64-60e_wflw-256x256.py) |  256x256   |         4.06         |         6.97         |             3.99             |           4.83            |         4.58         |          3.94          |            4.33            | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256-2bf032a6_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_wflw_256x256_20210125.log.json) |

<hr/>
<br/><br/>

## Cofw Dataset

<br/>

### Face 2d Keypoint + Hrnetv2 + Cofw on Cofw

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_iccv_2013/html/Burgos-Artizzu_Robust_Face_Landmark_2013_ICCV_paper.html">COFW (ICCV'2013)</a></summary>

```bibtex
@inproceedings{burgos2013robust,
  title={Robust face landmark estimation under occlusion},
  author={Burgos-Artizzu, Xavier P and Perona, Pietro and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={1513--1520},
  year={2013}
}
```

</details>

Results on COFW dataset

The model is trained on COFW train.

| Arch                                                           | Input Size | NME  |                              ckpt                              |                              log                               |
| :------------------------------------------------------------- | :--------: | :--: | :------------------------------------------------------------: | :------------------------------------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/cofw/td-hm_hrnetv2-w18_8xb64-60e_cofw-256x256.py) |  256x256   | 3.48 | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_cofw_256x256-49243ab8_20211019.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_cofw_256x256_20211019.log.json) |

<hr/>
<br/><br/>

## Aflw Dataset

<br/>

### Face 2d Keypoint + Hrnetv2 + Dark + Aflw on Aflw

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6130513/">AFLW (ICCVW'2011)</a></summary>

```bibtex
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```

</details>

Results on AFLW dataset

The model is trained on AFLW train and evaluated on AFLW full and frontal.

| Arch                                              | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub> |                       ckpt                        |                        log                        |
| :------------------------------------------------ | :--------: | :------------------: | :---------------------: | :-----------------------------------------------: | :-----------------------------------------------: |
| [pose_hrnetv2_w18_dark](/configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_dark-8xb64-60e_aflw-256x256.py) |  256x256   |         1.35         |          1.19           | [ckpt](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_aflw_256x256_dark-219606c0_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_aflw_256x256_dark_20210125.log.json) |

<br/>

### Face 2d Keypoint + Hrnetv2 + Aflw on Aflw

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6130513/">AFLW (ICCVW'2011)</a></summary>

```bibtex
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```

</details>

Results on AFLW dataset

The model is trained on AFLW train and evaluated on AFLW full and frontal.

| Arch                                              | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub> |                       ckpt                        |                        log                        |
| :------------------------------------------------ | :--------: | :------------------: | :---------------------: | :-----------------------------------------------: | :-----------------------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py) |  256x256   |         1.41         |          1.27           | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256_20210125.log.json) |
