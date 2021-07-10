# DeepPose: Human pose estimation via deep neural networks

## Introduction

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

DeepPose first proposes using deep neural networks (DNNs) to tackle the problem of pose estimation.
It follows the top-down paradigm, that first detects the bounding boxes and then estimates poses.
It learns to directly regress the face keypoint coordinates.
