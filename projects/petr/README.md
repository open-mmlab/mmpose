# YOLOX-Pose

This project implements PETR (Pose Estimation with TRansformers), an end-to-end multi-person pose estimation framework introduced in the CVPR 2022 paper **End-to-End Multi-Person Pose Estimation with Transformers**. PETR is a novel, end-to-end multi-person pose estimation method that treats pose estimation as a hierarchical set prediction problem. By leveraging attention mechanisms, PETR can adaptively focus on features most relevant to target keypoints, thereby overcoming feature misalignment issues in pose estimation.

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/ec7eb99d-8b8b-4c0d-9714-0ccd33a4f054" alt><br>

## Usage

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.6 or higher
- [MMEngine](https://github.com/open-mmlab/mmengine) v0.7.0 or higher
- [MMCV](https://github.com/open-mmlab/mmcv) v2.0.0 or higher
- [MMDetection](https://github.com/open-mmlab/mmdetection) v3.0.0 or higher
- [MMPose](https://github.com/open-mmlab/mmpose) v1.0.0 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. **In `petr/` root directory**, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Inference

Users can apply YOLOX-Pose models to estimate human poses using the inferencer found in the MMPose core package. Use the command below:

```shell
python demo/inferencer_demo.py $INPUTS \
    --pose2d $CONFIG --pose2d-weights $CHECKPOINT --scope mmdet \
    [--show] [--vis-out-dir $VIS_OUT_DIR] [--pred-out-dir $PRED_OUT_DIR]
```

For more information on using the inferencer, please see [this document](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html#out-of-the-box-inferencer).

### Results

Results on COCO val2017

| Model | Backbone | Lr schd | mAP  | AP<sup>50</sup> | AP<sup>75</sup> |  AR  | AR<sup>50</sup> |                      Config                      |                                    Download                                     |
| :---: | :------: | :-----: | :--: | :-------------: | :-------------: | :--: | :-------------: | :----------------------------------------------: | :-----------------------------------------------------------------------------: |
| PETR  |   R-50   |  100e   | 68.7 |      87.5       |      76.2       | 75.9 |      92.1       |  [config](/configs/petr_r50_8xb4-100e_coco.py)   | [Google Drive](https://drive.google.com/file/d/1HcwraqWdZ3CaGMQOJHY8exNem7UnFkfS/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1C0HbQWV7K-GHQE7q34nUZw?pwd=u798) |
| PETR  |  R-101   |  100e   | 70.0 |      88.5       |      77.5       | 77.0 |      92.6       |  [config](/configs/petr_r101_8xb4-100e_coco.py)  | [Google Drive](https://drive.google.com/file/d/1O261Jrt4JRGlIKTmLtPy3AUruwX1hsDf/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1D5wqNP53KNOKKE5NnO2Dnw?pwd=keyn) |
| PETR  |  Swin-L  |  100e   | 73.0 |      90.7       |      80.9       | 80.1 |      94.5       | [config](/configs/petr_swin-l_8xb4-100e_coco.py) | [Google Drive](https://drive.google.com/file/d/1ujL0Gm5tPjweT0-gdDGkTc7xXrEt6gBP/view?usp=sharing) \| [BaiduYun](https://pan.baidu.com/s/1X5Cdq75GosRCKqbHZTSpJQ?pwd=t9ea) |

Currently, the PETR implemented in this project supports inference using the official checkpoint. However, the training accuracy is still not up to the results reported in the paper. We will continue to update this project after aligning the training accuracy.

## Citation

If this project benefits your work, please kindly consider citing the original papers:

```bibtex
@inproceedings{shi2022end,
  title={End-to-end multi-person pose estimation with transformers},
  author={Shi, Dahu and Wei, Xing and Li, Liangqi and Ren, Ye and Tan, Wenming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11069--11078},
  year={2022}
}
```
