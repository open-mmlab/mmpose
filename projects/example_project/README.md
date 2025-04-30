# Example Project

> A README.md template for releasing a project.
>
> All the fields in this README are **mandatory** for others to understand what you have achieved in this implementation.
> Please read our [Projects FAQ](../faq.md) if you still feel unclear about the requirements, or raise an [issue](https://github.com/open-mmlab/mmpose/issues) to us!

## Description

> Share any information you would like others to know. For example:
>
> Author: @xxx.
>
> This is an implementation of \[XXX\].

Author： @xxx.

This project implements a top-down pose estimator with custom head and loss functions that have been seamlessly inherited from existing modules within MMPose.

## Usage

> For a typical model, this section should contain the commands for training and testing.
> You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`.

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim) v0.33 or higher
- [MMPose](https://github.com/open-mmlab/mmpose) v1.0.0rc0 or higher

All the commands below rely on the correct configuration of `PYTHONPATH`, which should point to the project's directory so that Python can locate the module files. In `example_project/` root directory, run the following line to add the current directory to `PYTHONPATH`:

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the COCO dataset according to the [instruction](https://mmpose.readthedocs.io/en/dev-1.x/dataset_zoo/2d_body_keypoint.html#coco).

### Training commands

**To train with single GPU:**

```shell
mim train mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py
```

**To train with multiple GPUs:**

```shell
mim train mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```shell
mim train mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```shell
mim test mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py $CHECKPOINT
```

**To test with multiple GPUs:**

```shell
mim test mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```shell
mim test mmpose configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py $CHECKPOINT --launcher slurm \
    --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

## Results

> List the results as usually done in other model's README. Here is an [Example](https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_coco.md).

> You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project

|                             Model                             | Backbone  | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                             Download                              |
| :-----------------------------------------------------------: | :-------: | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :---------------------------------------------------------------: |
| [ExampleHead + ExampleLoss](./configs/example-head-loss_hrnet-w32_8xb64-210e_coco-256x192.py) | HRNet-w32 |  256x912   | 0.749 |      0.906      |      0.821      | 0.804 |      0.945      | [model](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth) \| [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192_20220909.log) |

## Citation

> You may remove this section if not applicable.

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

> The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

> OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

> Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.

> A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    > The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmpose.registry.MODELS` and configurable via a config file.

  - [ ] Basic docstrings & proper citation

    > Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd)

  - [ ] Test-time correctness

    > If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone.

  - [ ] A full README

    > As this template does.

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

    > If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range.

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    > Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmpose/blob/0fb7f22000197181dc0629f767dd99d881d23d76/mmpose/utils/tensor_utils.py#L53)

  - [ ] Unit tests

    > Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmpose/blob/dev-1.x/tests/test_models/test_heads/test_heatmap_heads/test_heatmap_head.py)

  - [ ] Code polishing

    > Refactor your code according to reviewer's comment.

  - [ ] Metafile.yml

    > It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_coco.yml)

  - [ ] Move your modules into the core package following the codebase's file hierarchy structure.

    > In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/body_2d_keypoint/topdown_heatmap/README.md)

  - [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
