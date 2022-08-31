# Hand 2D Keypoint

<hr/>
<br/><br/>

## Onehand10k Dataset

<br/>

### Hand 2d Keypoint + Resnet + Onehand10k on Onehand10k

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

Results on OneHand10K val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  |  EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [deeppose_resnet_50](/configs/hand_2d_keypoint/topdown_regression/onehand10k/td-reg_res50_8xb64-210e_onehand10k-256x256.py) |  256x256   |  0.990  | 0.485 | 34.21 | [ckpt](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_onehand10k_256x256_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Mobilenetv2 + Onehand10k on Onehand10k

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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

Results on OneHand10K val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  |  EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_mobilenet_v2](/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_mobilenetv2_8xb64-210e_onehand10k-256x256.py) |  256x256   |  0.986  | 0.537 | 28.56 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_onehand10k_256x256-f3a3d90e_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_onehand10k_256x256_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Dark + Onehand10k on Onehand10k

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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

Results on OneHand10K val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  |  EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_hrnetv2_w18_dark](/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_dark-8xb32-210e_onehand10k-256x256.py) |  256x256   |  0.990  | 0.572 | 23.96 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Onehand10k on Onehand10k

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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

Results on OneHand10K val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  |  EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_hrnetv2_w18](/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb32-210e_onehand10k-256x256.py) |  256x256   |  0.990  | 0.567 | 24.26 | [ckpt](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Udp + Onehand10k on Onehand10k

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8529221/">OneHand10K (TCSVT'2019)</a></summary>

```bibtex
@article{wang2018mask,
  title={Mask-pose cascaded cnn for 2d hand pose estimation from single color image},
  author={Wang, Yangang and Peng, Cong and Liu, Yebin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={29},
  number={11},
  pages={3258--3268},
  year={2018},
  publisher={IEEE}
}
```

</details>

Results on OneHand10K val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  |  EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_hrnetv2_w18_udp](/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_udp-8xb32-210e_onehand10k-256x256.py) |  256x256   |  0.990  | 0.571 | 23.88 | [ckpt](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_onehand10k_256x256_udp_20210330.log.json) |

<hr/>
<br/><br/>

## Coco_wholebody_hand Dataset

<br/>

### Hand 2d Keypoint + Mobilenetv2 + Coco + Wholebody + Hand on Coco_wholebody_hand

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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

|                            Arch                            | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------: | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_mobilenetv2](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_mobilenetv2_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.795  | 0.829 | 4.77 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_coco_wholebody_hand_256x256-06b8c877_20210909.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_coco_wholebody_hand_256x256_20210909.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Coco + Wholebody + Hand on Coco_wholebody_hand

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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_hrnetv2_w18](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hrnetv2-w18_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.813  | 0.840 | 4.39 | [ckpt](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256-1c028db7_20210908.pth) | [log](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_coco_wholebody_hand_256x256_20210908.log.json) |

<br/>

### Hand 2d Keypoint + Litehrnet + Coco + Wholebody + Hand on Coco_wholebody_hand

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.06403">LiteHRNet (CVPR'2021)</a></summary>

```bibtex
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [LiteHRNet-18](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_litehrnet-w18_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.795  | 0.830 | 4.77 | [ckpt](https://download.openmmlab.com/mmpose/hand/litehrnet/litehrnet_w18_coco_wholebody_hand_256x256-d6945e6a_20210908.pth) | [log](https://download.openmmlab.com/mmpose/hand/litehrnet/litehrnet_w18_coco_wholebody_hand_256x256_20210908.log.json) |

<br/>

### Hand 2d Keypoint + Resnet + Coco + Wholebody + Hand on Coco_wholebody_hand

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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

|                            Arch                            | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------: | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_resnet_50](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_res50_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.800  | 0.833 | 4.64 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_coco_wholebody_hand_256x256-8dbc750c_20210908.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_coco_wholebody_hand_256x256_20210908.log.json) |

<br/>

### Hand 2d Keypoint + Hourglass + Coco + Wholebody + Hand on Coco_wholebody_hand

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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_hourglass_52](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hourglass52_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.804  | 0.835 | 4.54 | [ckpt](https://download.openmmlab.com/mmpose/hand/hourglass/hourglass52_coco_wholebody_hand_256x256-7b05c6db_20210909.pth) | [log](https://download.openmmlab.com/mmpose/hand/hourglass/hourglass52_coco_wholebody_hand_256x256_20210909.log.json) |

<br/>

### Hand 2d Keypoint + Scnet + Coco + Wholebody + Hand on Coco_wholebody_hand

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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

|                            Arch                            | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------: | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_scnet_50](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_scnet50_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.803  | 0.834 | 4.55 | [ckpt](https://download.openmmlab.com/mmpose/hand/scnet/scnet50_coco_wholebody_hand_256x256-e73414c7_20210909.pth) | [log](https://download.openmmlab.com/mmpose/hand/scnet/scnet50_coco_wholebody_hand_256x256_20210909.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Dark + Coco + Wholebody + Hand on Coco_wholebody_hand

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
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_12">COCO-WholeBody-Hand (ECCV'2020)</a></summary>

```bibtex
@inproceedings{jin2020whole,
  title={Whole-Body Human Pose Estimation in the Wild},
  author={Jin, Sheng and Xu, Lumin and Xu, Jin and Wang, Can and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

</details>

Results on COCO-WholeBody-Hand val set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_hrnetv2_w18_dark](/configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_hrnetv2-w18_dark-8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.814  | 0.840 | 4.37 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark_20210908.log.json) |

<hr/>
<br/><br/>

## Rhd2d Dataset

<br/>

### Hand 2d Keypoint + Hrnetv2 + Dark + Rhd2d on Rhd2d

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
<summary align="right"><a href="https://lmb.informatik.uni-freiburg.de/projects/hand3d/">RHD (ICCV'2017)</a></summary>

```bibtex
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

</details>

Results on RHD test set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_hrnetv2_w18_dark](/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py) |  256x256   |  0.992  | 0.903 | 2.18 | [ckpt](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_rhd2d_256x256_dark_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Resnet + Rhd2d on Rhd2d

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
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
<summary align="right"><a href="https://lmb.informatik.uni-freiburg.de/projects/hand3d/">RHD (ICCV'2017)</a></summary>

```bibtex
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

</details>

Results on RHD test set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [deeppose_resnet_50](/configs/hand_2d_keypoint/topdown_regression/rhd2d/td-reg_res50_8xb64-210e_rhd2d-256x256.py) |  256x256   |  0.988  | 0.865 | 3.32 | [ckpt](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_rhd2d_256x256-37f1c4d3_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_rhd2d_256x256_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Udp + Rhd2d on Rhd2d

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html">UDP (CVPR'2020)</a></summary>

```bibtex
@InProceedings{Huang_2020_CVPR,
  author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
  title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://lmb.informatik.uni-freiburg.de/projects/hand3d/">RHD (ICCV'2017)</a></summary>

```bibtex
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

</details>

Results on RHD test set

| Arch                                                       | Input Size | PCKh@0.7 |  AUC  | EPE  |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :--------: | :------: | :---: | :--: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [pose_hrnetv2_w18_udp](/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_udp-8xb64-210e_rhd2d-256x256.pyy) |  256x256   |  0.992   | 0.902 | 2.19 | [ckpt](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_rhd2d_256x256_udp-63ba6007_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_rhd2d_256x256_udp_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Hrnetv2 + Rhd2d on Rhd2d

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
<summary align="right"><a href="https://lmb.informatik.uni-freiburg.de/projects/hand3d/">RHD (ICCV'2017)</a></summary>

```bibtex
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

</details>

Results on RHD test set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_hrnetv2_w18](/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_8xb64-210e_rhd2d-256x256.py) |  256x256   |  0.992  | 0.902 | 2.21 | [ckpt](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256-95b20dd8_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_rhd2d_256x256_20210330.log.json) |

<br/>

### Hand 2d Keypoint + Mobilenetv2 + Rhd2d on Rhd2d

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
<summary align="right"><a href="https://lmb.informatik.uni-freiburg.de/projects/hand3d/">RHD (ICCV'2017)</a></summary>

```bibtex
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

</details>

Results on RHD test set

| Arch                                                       | Input Size | PCK@0.2 |  AUC  | EPE  |                            ckpt                            |                            log                             |
| :--------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :--------------------------------------------------------: | :--------------------------------------------------------: |
| [pose_mobilenet_v2](/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_mobilenetv2_8xb64-210e_rhd2d-256x256.py) |  256x256   |  0.985  | 0.883 | 2.79 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_rhd2d_256x256-85fa02db_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_rhd2d_256x256_20210330.log.json) |

<hr/>
<br/><br/>

## Freihand2d Dataset

<br/>

### Hand 2d Keypoint + Resnet + Freihand2d on Freihand2d

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
<summary align="right"><a href="http://openaccess.thecvf.com/content_ICCV_2019/html/Zimmermann_FreiHAND_A_Dataset_for_Markerless_Capture_of_Hand_Pose_and_ICCV_2019_paper.html">FreiHand (ICCV'2019)</a></summary>

```bibtex
@inproceedings{zimmermann2019freihand,
  title={Freihand: A dataset for markerless capture of hand pose and shape from single rgb images},
  author={Zimmermann, Christian and Ceylan, Duygu and Yang, Jimei and Russell, Bryan and Argus, Max and Brox, Thomas},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={813--822},
  year={2019}
}
```

</details>

Results on FreiHand val & test set

| Set  |                           Arch                            | Input Size | PCK@0.2 |  AUC  | EPE  |                           ckpt                            |                           log                            |
| :--- | :-------------------------------------------------------: | :--------: | :-----: | :---: | :--: | :-------------------------------------------------------: | :------------------------------------------------------: |
| test | [pose_resnet_50](/configs/hand_2d_keypoint/topdown_heatmap/freihand2d/td-hm_res50_8xb64-100e_freihand2d-224x224.py) |  224x224   |  0.999  | 0.868 | 3.27 | [ckpt](https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224-ff0799bc_20200914.pth) | [log](https://download.openmmlab.com/mmpose/hand/resnet/res50_freihand_224x224_20200914.log.json) |
