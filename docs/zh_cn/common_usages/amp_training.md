# 自动混合精度（AMP）训练

混合精度训练在不改变模型、不降低模型训练精度的前提下，可以缩短训练时间，降低存储需求，因而能支持更大的 batch size、更大模型和尺寸更大的输入的训练。

如果要开启自动混合精度（AMP）训练，在训练命令最后加上 --amp 即可， 命令如下：

```
python tools/train.py ${CONFIG_FILE} --amp
```

具体例子如下：

```
python tools/train.py configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py  --amp
```
