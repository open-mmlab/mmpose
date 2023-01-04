# Example Project

This is an example README for community `projects/`. You can write your README in your own project. Here are
some recommended parts of a README for others to understand and use your project, you can copy or modify them
according to your project.

## Usage

### Setup Environment

Please refer to [Installation](https://mmpose.readthedocs.io/en/latest/install.html#installation) to install MMPose.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the COCO dataset according to the [instruction](https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html#coco).

### Training commands

**To train with single GPU:**

```bash
mim train mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py
```

**To train with multiple GPUs:**

```bash
mim train mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py $CHECKPOINT --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results

|           Model           | Backbone  |  AP   |  AR   |                                   Config                                   |                                      Download                                      |
| :-----------------------: | :-------: | :---: | :---: | :------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| ExampleHead + ExampleLoss | HRNet-w32 | 82.33 | 96.15 | [config](./configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py) | [model](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) \| [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_20220909.log) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

## Checklist

Here is a checklist of this project's progress. And you can ignore this part if you don't plan to contribute
to MMPose projects.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmpose.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. Usually, ±0.1% mAP is acceptable for the keypoint detections task on COCO. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmpose/blob/1.x/tests/test_models/test_heads/test_heatmap_heads/test_heatmap_head.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for MMPose to acquire your models. [Example](https://github.com/open-mmlab/mmpose/blob/1.x/configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_coco.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmpose/blob/1.x/configs/body_2d_keypoint/topdown_heatmap/README.md) -->
