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

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [LiteHRNet-18](/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/litehrnet_w18_coco_wholebody_hand_256x256.py) | 256x256 | 0.795 | 0.830 | 4.77 | [ckpt](https://download.openmmlab.com/mmpose/hand/litehrnet/litehrnet_w18_coco_wholebody_hand_256x256-d6945e6a_20210908.pth) | [log](https://download.openmmlab.com/mmpose/hand/litehrnet/litehrnet_w18_coco_wholebody_hand_256x256_20210908.log.json) |
