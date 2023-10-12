# YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2204.06806">YOLO-Pose (CVPRW'2022)</a></summary>

```bibtex
@inproceedings{maji2022yolo,
  title={Yolo-pose: Enhancing yolo for multi person pose estimation using object keypoint similarity loss},
  author={Maji, Debapriya and Nagori, Soyeb and Mathew, Manu and Poddar, Deepak},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2637--2646},
  year={2022}
}
```

</details>

YOLO-Pose is a bottom-up pose estimation approach that simultaneously detects all person instances and regresses keypoint locations in a single pass.

We implement **YOLOX-Pose** based on the **YOLOX** object detection framework and inherits the benefits of unified pose estimation and object detection from YOLO-pose. To predict keypoint locations more accurately, separate branches with adaptive convolutions are used to regress the offsets for different joints. This allows optimizing the feature extraction for each keypoint.
