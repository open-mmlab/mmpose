<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{Zhou_2023_CVPR,
    author    = {Zhou, Zhenglin and Li, Huaxia and Liu, Hong and Wang, Nanyang and Yu, Gang and Ji, Rongrong},
    title     = {STAR Loss: Reducing Semantic Ambiguity in Facial Landmark Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15475-15484}
}
```

</details>

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_STAR_Loss_Reducing_Semantic_Ambiguity_in_Facial_Landmark_Detection_CVPR_2023_paper.pdf">STARloss (CVPR'2023)</a></summary>

```bibtex
@article{lin2021structure,
  title={Structure-Coherent Deep Feature Learning for Robust Face Alignment},
  author={Lin, Chunze and Zhu, Beier and Wang, Quan and Liao, Renjie and Qian, Chen and Lu, Jiwen and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  year={2021},
  publisher={IEEE}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Look_at_Boundary_CVPR_2018_paper.html">WFLW (CVPR'2018)</a></summary>

```bibtex
@inproceedings{wu2018look,
  title={Look at boundary: A boundary-aware face alignment algorithm},
  author={Wu, Wayne and Qian, Chen and Yang, Shuo and Wang, Quan and Cai, Yici and Zhou, Qiang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2129--2138},
  year={2018}
}
```

</details>

Results on WFLW dataset

The model is trained on WFLW train set.

| Model                                                                                                                    | Input Size | NME |    ckpt    |    log    |
| :----------------------------------------------------------------------------------------------------------------------- | :--------: | :-: | :--------: | :-------: |
| [ResNet-50+STARLoss](/configs/face_2d_keypoint/topdown_regression/wflw/td-reg_res50_starloss_8xb64-210e_wflw-256x256.py) |  256x256   |     | [ckpt](<>) | [log](<>) |
