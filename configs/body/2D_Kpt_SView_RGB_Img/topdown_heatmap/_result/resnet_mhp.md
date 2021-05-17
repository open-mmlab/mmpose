<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{zhao2018understanding,
  title={Understanding humans in crowded scenes: Deep nested adversarial learning and a new benchmark for multi-human parsing},
  author={Zhao, Jian and Li, Jianshu and Cheng, Yu and Sim, Terence and Yan, Shuicheng and Feng, Jiashi},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={792--800},
  year={2018}
}
```

#### Results on MHP v2.0 val set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_101](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mhp/res50_mhp_256x192.py) | 256x192 | 0.583 | 0.897 | 0.669 | 0.636 | 0.918 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mhp_256x192-28c5b818_20201229.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mhp_256x192_20201229.log.json) |

Note that, the evaluation metric used here is mAP (adapted from COCO), which may be different from the official evaluation [codes](https://github.com/ZhaoJ9014/Multi-Human-Parsing/tree/master/Evaluation/Multi-Human-Pose).
Please be cautious if you use the results in papers.
