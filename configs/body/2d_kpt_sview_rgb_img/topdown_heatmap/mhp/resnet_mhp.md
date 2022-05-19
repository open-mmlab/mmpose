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
<summary align="right"><a href="https://dl.acm.org/doi/abs/10.1145/3240508.3240509">MHP (ACM MM'2018)</a></summary>

```bibtex
@inproceedings{zhao2018understanding,
  title={Understanding humans in crowded scenes: Deep nested adversarial learning and a new benchmark for multi-human parsing},
  author={Zhao, Jian and Li, Jianshu and Cheng, Yu and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={792--800},
  year={2018}
}
```

</details>

Results on MHP v2.0 val set

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mhp/res50_mhp_256x192.py) |  256x192   | 0.583 |      0.897      |      0.669      | 0.636 |      0.918      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mhp_256x192-28c5b818_20201229.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mhp_256x192_20201229.log.json) |

Note that, the evaluation metric used here is mAP (adapted from COCO), which may be different from the official evaluation [codes](https://github.com/ZhaoJ9014/Multi-Human-Parsing/tree/master/Evaluation/Multi-Human-Pose).
Please be cautious if you use the results in papers.
