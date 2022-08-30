<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/9052469/">HRNetv2 (TPAMI'2019)</a></summary>

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://www.sciencedirect.com/science/article/pii/S0262885616000147">300W (IMAVIS'2016)</a></summary>

```bibtex
@article{sagonas2016300,
  title={300 faces in-the-wild challenge: Database and results},
  author={Sagonas, Christos and Antonakos, Epameinondas and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  journal={Image and vision computing},
  volume={47},
  pages={3--18},
  year={2016},
  publisher={Elsevier}
}
```

</details>

Results on 300W dataset

The model is trained on 300W train.

| Arch                               | Input Size | NME<sub>*common*</sub> | NME<sub>*challenge*</sub> | NME<sub>*full*</sub> | NME<sub>*test*</sub> |                ckpt                 |                log                 |
| :--------------------------------- | :--------: | :--------------------: | :-----------------------: | :------------------: | :------------------: | :---------------------------------: | :--------------------------------: |
| [pose_hrnetv2_w18](/configs/face_2d_keypoint/topdown_heatmap/300w/td-hm_hrnetv2-w18_8xb64-60e_300w-256x256.py) |  256x256   |          2.92          |           5.64            |         3.45         |         4.10         | [ckpt](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256-eea53406_20211019.pth) | [log](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_300w_256x256_20211019.log.json) |
