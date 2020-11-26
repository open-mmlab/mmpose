# End-to-end Recovery of Human Shape and Pose

## Introduction
```
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

## Results and models

### 3D Human Mesh Estimation

#### Results on Human3.6M with ground-truth bounding box having MPJPE-PA of 52.60 mm on Protocol2

| Arch  | Input Size | MPJPE (P1)| MPJPE-PA (P1) | MPJPE (P2) | MPJPE-PA (P2) | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |
| [hmr_resnet_50](/configs/mesh/hmr/hmr_resnet_50.py)  | 224x224 | 80.75 | 55.08 | 80.35 | 52.60 | [ckpt](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224-c21e8229_20201015.pth) | [log](https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224_20201015.log.json) |
