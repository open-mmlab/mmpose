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

AE is one of the most popular 2D bottom-up pose estimation approaches, that first detect all the keypoints and
then group/associate them into person instances.

In order to group all the predicted keypoints to individuals, a tag is also predicted for each detected keypoint.
Tags of the same person are similar, while tags of different people are different. Thus the keypoints can be grouped
according to the tags.
