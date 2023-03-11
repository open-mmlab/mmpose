# Automatic Mixed Precision (AMP) Training

Mixed precision training can reduce training time and storage requirements without changing the model or reducing the model training accuracy, thus supporting larger batch sizes, larger models, and larger input sizes.

To enable Automatic Mixing Precision (AMP) training, add `--amp` to the end of the training command, which is as follows:

```
python tools/train.py ${CONFIG_FILE} --amp
```

Specific examples are as follows:

```
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py  --amp
```
