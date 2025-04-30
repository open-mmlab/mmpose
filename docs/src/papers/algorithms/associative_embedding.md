# Associative embedding: End-to-end learning for joint detection and grouping (AE)

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.05424">Associative Embedding (NIPS'2017)</a></summary>

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

We introduce associative embedding, a novel method for supervising convolutional neural networks for the task of detection and grouping. A number of computer vision problems can be framed in this manner including multi-person pose estimation, instance segmentation, and multi-object tracking. Usually the grouping of detections is achieved with multi-stage pipelines, instead we propose an approach that teaches a network to simultaneously output detections and group assignments. This technique can be easily integrated into any state-of-the-art network architecture that produces pixel-wise predictions. We show how to apply this method to both multi-person pose estimation and instance segmentation and report state-of-the-art performance for multi-person pose on the MPII and MS-COCO datasets.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146514181-84f22623-6b73-4656-89b8-9e7f551e9cc0.png">
</div>
