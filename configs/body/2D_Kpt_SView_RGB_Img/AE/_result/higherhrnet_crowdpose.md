<!-- [ALGORITHM] -->

```bibtex
@inproceedings{cheng2020higherhrnet,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5386--5395},
  year={2020}
}
```

<!-- [DATASET] -->

```bibtex
@article{li2018crowdpose,
  title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
  author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
  journal={arXiv preprint arXiv:1812.00324},
  year={2018}
}
```

#### Results on CrowdPose test without multi-scale test

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [HigherHRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/crowdpose/higher_hrnet32_crowdpose_512x512.py)  | 512x512 | 0.655 | 0.859 | 0.705 | 0.728 | 0.660 | 0.577 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512-1aa4a132_20201017.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512_20201017.log.json) |

#### Results on CrowdPose test with multi-scale test. 2 scales (\[2, 1\]) are used

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [HigherHRNet-w32](/configs/body/2D_Kpt_SView_RGB_Img/AE/crowdpose/higher_hrnet32_crowdpose_512x512.py)  | 512x512 | 0.661 | 0.864 | 0.710 | 0.742 | 0.670 | 0.566 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512-1aa4a132_20201017.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_crowdpose_512x512_20201017.log.json) |
