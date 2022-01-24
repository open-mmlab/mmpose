# Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/pdf/1904.07399.pdf">AdaptiveWingloss (ICCV'2019)</a></summary>

```bibtex
@inproceedings{wang2019adaptive,
  title={Adaptive wing loss for robust face alignment via heatmap regression},
  author={Wang, Xinyao and Bo, Liefeng and Fuxin, Li},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6971--6981},
  year={2019}
}
```

</details>

## Abstract

<!-- [ABSTRACT] -->

Heatmap regression with a deep network has become one of the mainstream approaches to localize facial landmarks. However, the loss function for heatmap regression is rarely studied. In this paper, we analyze the ideal loss function properties for heatmap regression in face alignment problems. Then we propose a novel loss function, named Adaptive Wing loss, that is able to adapt its shape to different types of ground truth heatmap pixels. This adaptability penalizes loss more on foreground pixels while less on background pixels. To address the imbalance between foreground and background pixels, we also propose Weighted Loss Map, which assigns high weights on foreground and difficult background pixels to help training process focus more on pixels that are crucial to landmark localization. To further improve face alignment accuracy, we introduce boundary prediction and CoordConv with boundary coordinates. Extensive experiments on different benchmarks, including COFW, 300W and WFLW, show our approach outperforms the state-of-the-art by a significant margin on
various evaluation metrics. Besides, the Adaptive Wing loss also helps other heatmap regression tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/148007960-a06a34d8-8090-49e1-80db-6bbe4a7e7e8d.png">
</div>
