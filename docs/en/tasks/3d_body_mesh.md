# 3D Body Mesh Recovery Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

To achieve high-quality human mesh estimation, we use multiple datasets for training.
The following items should be prepared for human mesh training:

<!-- TOC -->

- [3D Body Mesh Recovery Datasets](#3d-body-mesh-recovery-datasets)
  - [Notes](#notes)
    - [Annotation Files for Human Mesh Estimation](#annotation-files-for-human-mesh-estimation)
    - [SMPL Model](#smpl-model)
  - [COCO](#coco)
  - [Human3.6M](#human36m)
  - [MPI-INF-3DHP](#mpi-inf-3dhp)
  - [LSP](#lsp)
  - [LSPET](#lspet)
  - [CMU MoShed Data](#cmu-moshed-data)

<!-- TOC -->

## Notes

### Annotation Files for Human Mesh Estimation

For human mesh estimation, we use multiple datasets for training.
The annotation of different datasets are preprocessed to the same format. Please
follow the [preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to generate the annotation files or download the processed files from
[here](https://download.openmmlab.com/mmpose/datasets/mesh_annotation_files.zip),
and make it look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mesh_annotation_files
        ├── coco_2014_train.npz
        ├── h36m_valid_protocol1.npz
        ├── h36m_valid_protocol2.npz
        ├── hr-lspet_train.npz
        ├── lsp_dataset_original_train.npz
        ├── mpi_inf_3dhp_train.npz
        └── mpii_train.npz
```

### SMPL Model

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://dl.acm.org/doi/abs/10.1145/2816795.2818013">SMPL (TOG'2015)</a></summary>

```bibtex
@article{loper2015smpl,
  title={SMPL: A skinned multi-person linear model},
  author={Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J},
  journal={ACM transactions on graphics (TOG)},
  volume={34},
  number={6},
  pages={1--16},
  year={2015},
  publisher={ACM New York, NY, USA}
}
```

</details>

For human mesh estimation, SMPL model is used to generate the human mesh.
Please download the [gender neutral SMPL model](http://smplify.is.tue.mpg.de/),
[joints regressor](https://download.openmmlab.com/mmpose/datasets/joints_regressor_cmr.npy)
and [mean parameters](https://download.openmmlab.com/mmpose/datasets/smpl_mean_params.npz)
under `$MMPOSE/models/smpl`, and make it look like this:

```text
mmpose
├── mmpose
├── ...
├── models
    │── smpl
        ├── joints_regressor_cmr.npy
        ├── smpl_mean_params.npz
        └── SMPL_NEUTRAL.pkl
```

## COCO

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

For [COCO](http://cocodataset.org/) data, please download from [COCO download](http://cocodataset.org/#download). COCO'2014 Train is needed for human mesh estimation training.
Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── coco
        │-- train2014
        │   ├── COCO_train2014_000000000009.jpg
        │   ├── COCO_train2014_000000000025.jpg
        │   ├── COCO_train2014_000000000030.jpg
        |   │-- ...

```

## Human3.6M

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php), we use the MoShed data provided in [HMR](https://github.com/akanazawa/hmr) for training.
However, due to license limitations, we are not allowed to redistribute the MoShed data.

For the evaluation on Human3.6M dataset, please follow the
[preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to extract test images from
[Human3.6M](http://vision.imar.ro/human3.6m/description.php) original videos,
and make it look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── Human3.6M
        ├── images
            ├── S11_Directions_1.54138969_000001.jpg
            ├── S11_Directions_1.54138969_000006.jpg
            ├── S11_Directions_1.54138969_000011.jpg
            ├── ...
```

The download of Human3.6M dataset is quite difficult, you can also download the
[zip file](https://drive.google.com/file/d/1WnRJD9FS3NUf7MllwgLRJJC-JgYFr8oi/view?usp=sharing)
of the test images. However, due to the license limitations, we are not allowed to
redistribute the images either. So the users need to download the original video and
extract the images by themselves.

## MPI-INF-3DHP

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8374605/">MPI-INF-3DHP (3DV'2017)</a></summary>

```bibtex
@inproceedings{mono-3dhp2017,
  author = {Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
  title = {Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision},
  booktitle = {3D Vision (3DV), 2017 Fifth International Conference on},
  url = {http://gvv.mpi-inf.mpg.de/3dhp_dataset},
  year = {2017},
  organization={IEEE},
  doi={10.1109/3dv.2017.00064},
}
```

</details>

For [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/), please follow the
[preprocess procedure](https://github.com/nkolot/SPIN/tree/master/datasets/preprocess)
of SPIN to sample images, and make them like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── mpi_inf_3dhp_test_set
    │   ├── TS1
    │   ├── TS2
    │   ├── TS3
    │   ├── TS4
    │   ├── TS5
    │   └── TS6
    ├── S1
    │   ├── Seq1
    │   └── Seq2
    ├── S2
    │   ├── Seq1
    │   └── Seq2
    ├── S3
    │   ├── Seq1
    │   └── Seq2
    ├── S4
    │   ├── Seq1
    │   └── Seq2
    ├── S5
    │   ├── Seq1
    │   └── Seq2
    ├── S6
    │   ├── Seq1
    │   └── Seq2
    ├── S7
    │   ├── Seq1
    │   └── Seq2
    └── S8
        ├── Seq1
        └── Seq2
```

## LSP

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://www.bmva.org/bmvc/2010/conference/paper12/paper12.pdf">LSP (BMVC'2010)</a></summary>

```bibtex
@inproceedings{johnson2010clustered,
  title={Clustered Pose and Nonlinear Appearance Models for Human Pose Estimation.},
  author={Johnson, Sam and Everingham, Mark},
  booktitle={bmvc},
  volume={2},
  number={4},
  pages={5},
  year={2010},
  organization={Citeseer}
}
```

</details>

For [LSP](https://sam.johnson.io/research/lsp.html), please download the high resolution version
[LSP dataset original](http://sam.johnson.io/research/lsp_dataset_original.zip).
Extract them under `$MMPOSE/data`, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── lsp_dataset_original
        ├── images
            ├── im0001.jpg
            ├── im0002.jpg
            └── ...
```

## LSPET

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/5995318/">LSPET (CVPR'2011)</a></summary>

```bibtex
@inproceedings{johnson2011learning,
  title={Learning effective human pose estimation from inaccurate annotation},
  author={Johnson, Sam and Everingham, Mark},
  booktitle={CVPR 2011},
  pages={1465--1472},
  year={2011},
  organization={IEEE}
}
```

</details>

For [LSPET](https://sam.johnson.io/research/lspet.html), please download its high resolution form
[HR-LSPET](http://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip).
Extract them under `$MMPOSE/data`, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── lspet_dataset
        ├── images
        │   ├── im00001.jpg
        │   ├── im00002.jpg
        │   ├── im00003.jpg
        │   └── ...
        └── joints.mat
```

## CMU MoShed Data

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf">CMU MoShed (CVPR'2018)</a></summary>

```bibtex
@inproceedings{kanazawa2018end,
  title={End-to-end recovery of human shape and pose},
  author={Kanazawa, Angjoo and Black, Michael J and Jacobs, David W and Malik, Jitendra},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7122--7131},
  year={2018}
}
```

</details>

Real-world SMPL parameters are used for the adversarial training in human mesh estimation.
The MoShed data provided in [HMR](https://github.com/akanazawa/hmr) is included in this
[zip file](https://download.openmmlab.com/mmpose/datasets/mesh_annotation_files.zip).
Please download and extract it under `$MMPOSE/data`, and make it look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── mesh_annotation_files
        ├── CMU_mosh.npz
        └── ...
```
