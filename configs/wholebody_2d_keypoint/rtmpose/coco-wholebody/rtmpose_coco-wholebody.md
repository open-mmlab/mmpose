<!-- [ALGORITHM] -->

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
| [rtmpose-m](./rtmpose-m_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |  0.687  |  0.734  |  0.640  |  0.733  |  0.821  |  0.856  |  0.459  |  0.542  |  0.591   |  0.656   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.json) |
| [rtmpose-l](./rtmpose-l_8xb64-270e_coco-wholebody-256x192.py) |  256x192   |  0.712  |  0.756  |  0.683  |  0.770  |  0.848  |  0.879  |  0.502  |  0.581  |  0.622   |  0.685   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.json) |
| [rtmpose-l](./rtmpose-l_8xb32-270e_coco-wholebody-384x288.py) |  384x288   |  0.728  |  0.769  |  0.727  |  0.800  |  0.892  |  0.915  |  0.572  |  0.643  |  0.661   |  0.716   | [ckpt](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth) | [log](https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.json) |
