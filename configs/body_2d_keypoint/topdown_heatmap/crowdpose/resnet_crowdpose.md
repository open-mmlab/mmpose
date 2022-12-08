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
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Li_CrowdPose_Efficient_Crowded_Scenes_Pose_Estimation_and_a_New_Benchmark_CVPR_2019_paper.html">CrowdPose (CVPR'2019)</a></summary>

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

</details>

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

| Arch                                           | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) |                      ckpt                      |                      log                      |
| :--------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :----: | :----: | :----: | :--------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_50](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res50_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.637 |      0.808      |      0.692      | 0.738  | 0.650  | 0.506  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192-c6a526b6_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192_20201227.log.json) |
| [pose_resnet_101](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res101_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.647 |      0.810      |      0.703      | 0.745  | 0.658  | 0.521  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192-8f5870f4_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192_20201227.log.json) |
| [pose_resnet_101](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res101_8xb64-210e_crowdpose-320x256.py) |  320x256   | 0.661 |      0.821      |      0.714      | 0.759  | 0.672  | 0.534  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256-c88c512a_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256_20201227.log.json) |
| [pose_resnet_152](/configs/body_2d_keypoint/topdown_heatmap/crowdpose/td-hm_res152_8xb64-210e_crowdpose-256x192.py) |  256x192   | 0.656 |      0.818      |      0.712      | 0.754  | 0.666  | 0.533  | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192-dbd49aba_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192_20201227.log.json) |
