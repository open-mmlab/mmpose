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
@inproceedings{duan2019trb,
  title={TRB: A Novel Triplet Representation for Understanding 2D Human Body},
  author={Duan, Haodong and Lin, Kwan-Yee and Jin, Sheng and Liu, Wentao and Qian, Chen and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9479--9488},
  year={2019}
}
```

#### Results on MPII-TRB val set

| Arch  | Input Size | Skeleton Acc   | Contour Acc   | Mean Acc | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |:------: |
| [pose_resnet_50](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii_trb/res50_mpii_trb_256x256.py)  | 256x256 | 0.887 | 0.858 | 0.868 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_trb_256x256-896036b8_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_trb_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii_trb/res101_mpii_trb_256x256.py)  | 256x256 | 0.890 | 0.863 | 0.873 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_trb_256x256-cfad2f05_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_trb_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii_trb/res152_mpii_trb_256x256.py)  | 256x256 | 0.897 | 0.868 | 0.879 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_trb_256x256-dd369ce6_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_trb_256x256_20200812.log.json) |
