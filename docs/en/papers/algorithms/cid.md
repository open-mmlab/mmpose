# Contextual Instance Decoupling for Robust Multi-Person Pose Estimation

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Contextual_Instance_Decoupling_for_Robust_Multi-Person_Pose_Estimation_CVPR_2022_paper.pdf">CID (CVPR'2022)</a></summary>

```bibtex
@inproceedings{wang2022contextual,
  title={Contextual Instance Decoupling for Robust Multi-Person Pose Estimation},
  author={Wang, Dongkai and Zhang, Shiliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11060--11068},
  year={2022}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Crowded scenes make it challenging to differentiate persons and locate their pose keypoints. This paper proposes the Contextual Instance Decoupling (CID), which presents a new pipeline for multi-person pose estimation. Instead of relying on person bounding boxes to spatially differentiate persons, CID decouples persons in an image into multiple instance-aware feature maps. Each of those feature maps is hence adopted to infer keypoints for a specific person. Compared with bounding box detection, CID is differentiable and robust to detection errors. Decoupling persons into different feature maps allows to isolate distractions from other persons, and explore context cues at scales larger than the bounding box size. Experiments show that CID outperforms previous multi-person pose estimation pipelines on crowded scenes pose estimation benchmarks in both accuracy and efficiency. For instance, it achieves 71.3% AP on CrowdPose, outperforming the recent single-stage DEKR by 5.6%, the bottom-up CenterAttention by 3.7%, and the top-down JCSPPE by 5.3%. This advantage sustains on the commonly used COCO benchmark.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/kennethwdk/CID/raw/main/img/framework.png">
</div>
