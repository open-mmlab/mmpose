# rtmlib

![demo](https://github.com/Tau-J/rtmlib/assets/13503330/b7e8ce8b-3134-43cf-bba6-d81656897289)

rtmlib is a super lightweight library to conduct pose estimation based on [RTMPose](https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose) models **WITHOUT** any dependencies like mmcv, mmpose, mmdet, etc.

Basically, rtmlib only requires these dependencies:

- numpy
- opencv-python
- opencv-contrib-python
- onnxruntime

Optionally, you can use other common backends like opencv, onnxruntime, openvino, tensorrt to accelerate the inference process.

- For openvino users, please add the path `<your python path>\envs\<your env name>\Lib\site-packages\openvino\libs` into your environment path.

## Installation

- install from pypi:

```shell
pip install rtmlib -i https://pypi.org/simple
```

- install from source code:

```shell
git clone https://github.com/Tau-J/rtmlib.git
cd rtmlib

pip install -r requirements.txt

pip install -e .

# [optional]
# pip install onnxruntime-gpu
# pip install openvino

```

## Quick Start

Run `webui.py`:

```shell
# Please make sure you have installed gradio
# pip install gradio

python webui.py
```

![image](https://github.com/Tau-J/rtmlib/assets/13503330/49ef11a1-a1b5-4a20-a2e1-d49f8be6a25d)

Here is also a simple demo to show how to use rtmlib to conduct pose estimation on a single image.

```python
import cv2

from rtmlib import Wholebody, draw_skeleton

device = 'cpu'  # cpu, cuda
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
img = cv2.imread('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)

keypoints, scores = wholebody(img)

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)


cv2.imshow('img', img_show)
cv2.waitKey()
```

### Visualization

|                                            MMPose-style                                             |                                            OpenPose-style                                             |
| :-------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/c9e6fbaa-00f0-4961-ac87-d881edca778b"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/9afc996a-59e6-4200-a655-59dae10b46c4"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/b12e5f60-fec0-42a1-b7b6-365e93894fb1"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/5acf7431-6ef0-44a8-ae52-9d8c8cb988c9"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/091b8ce3-32d5-463b-9f41-5c683afa7a11"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/4ffc7be1-50d6-44ff-8c6b-22ea8975aad4"> |
| <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/6fddfc14-7519-42eb-a7a4-98bf5441f324"> | <img width="357" alt="result" src="https://github.com/Tau-J/rtmlib/assets/13503330/2523e568-e0c3-4c2e-8e54-d1a67100c537"> |

### Citation

```
@misc{rtmlib,
  title={rtmlib},
  author={Jiang, Tao},
  year={2023},
  howpublished = {\url{https://github.com/Tau-J/rtmlib}},
}

@misc{jiang2023,
  doi = {10.48550/ARXIV.2303.07399},
  url = {https://arxiv.org/abs/2303.07399},
  author = {Jiang, Tao and Lu, Peng and Zhang, Li and Ma, Ningsheng and Han, Rui and Lyu, Chengqi and Li, Yining and Chen, Kai},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

@misc{lu2023rtmo,
      title={{RTMO}: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
      author={Peng Lu and Tao Jiang and Yining Li and Xiangtai Li and Kai Chen and Wenming Yang},
      year={2023},
      eprint={2312.07526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
