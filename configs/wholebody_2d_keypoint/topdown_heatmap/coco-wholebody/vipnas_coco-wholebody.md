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
| [S-ViPNAS-MobileNetV3](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-mbv3_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.619  |  0.700  |  0.477  |  0.608  |  0.585  |  0.689  |  0.386  |  0.505  |  0.473   |  0.578   | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192-0fee581a_20211205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_20211205.log.json) |
| [S-ViPNAS-Res50](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_8xb64-210e_coco-wholebody-256x192.py) |  256x192   |  0.643  |  0.726  |  0.553  |  0.694  |  0.587  |  0.698  |  0.410  |  0.529  |  0.495   |  0.607   | [ckpt](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192-49e1c3a4_20211112.pth) | [log](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_20211112.log.json) |
