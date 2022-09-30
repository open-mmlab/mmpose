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
| [pose_resnet_50](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res50_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.652  |  0.738  |  0.615  |  0.749  |  0.606  |  0.715  |  0.460  |  0.584  |  0.521   |  0.633   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192-9e37ed88_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_50](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res50_8xb64-210e_coco-wholebody-384x288.py) |  384x288   |  0.666  |  0.747  |  0.634  |  0.763  |  0.731  |  0.811  |  0.536  |  0.646  |  0.574   |  0.670   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288-ce11e294_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_101](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res101_8xb32-210e_coco-wholebody-256x192.py) |  256x192   |  0.669  |  0.753  |  0.637  |  0.766  |  0.611  |  0.722  |  0.463  |  0.589  |  0.531   |  0.645   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192-7325f982_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_101](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res101_8xb32-210e_coco-wholebody-384x288.py) |  384x288   |  0.692  |  0.770  |  0.680  |  0.799  |  0.746  |  0.820  |  0.548  |  0.657  |  0.597   |  0.693   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288-6c137b9a_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_152](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res152_8xb32-210e_coco-wholebody-256x192.py) |  256x192   |  0.682  |  0.764  |  0.661  |  0.787  |  0.623  |  0.728  |  0.481  |  0.607  |  0.548   |  0.661   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192-5de8ae23_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_152](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res152_8xb32-210e_coco-wholebody-384x288.py) |  384x288   |  0.704  |  0.780  |  0.693  |  0.813  |  0.751  |  0.824  |  0.559  |  0.666  |  0.610   |  0.705   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288-eab8caa8_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288_20201004.log.json) |
