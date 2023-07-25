# Just Dance - A Simple Implementation

<sup>
   <a href="https://openxlab.org.cn/apps/detail/mmpose/just_dance-mmpose">
      <i>Try it on OpenXLab</i>
   </a>
</sup>

This project presents a dance scoring system based on RTMPose. Users can compare the similarity between two dancers in different videos: one referred to as the "teacher video" and the other as the "student video."

Here are examples of the output dance comparison:

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/56d5c4d1-55d8-4222-b481-2418cc29a8d4" width="600"/>

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/f93b94c7-529f-4704-8246-c3c812f4c31a" width="600"/>

## Usage

### Jupyter Notebook

We provide a Jupyter Notebook [`just_dance_demo.ipynb`](./just_dance_demo.ipynb) that contains the complete process of dance comparison. It includes steps such as video FPS adjustment, pose estimation, snippet alignment, scoring, and the generation of the merged video.

### CLI tool

Users can simply run the following command to generate the comparison video:

```shell
python process_video.py ${TEACHER_VIDEO} ${STUDENT_VIDEO}
```

### Gradio

Users can also utilize Gradio to build an application using this system. We provide the script [`app.py`](./app.py). This application supports webcam input in addition to existing videos. To build this application, please follow these two steps:

1. Install Gradio
   ```shell
   pip install gradio
   ```
2. Run the script [`app.py`](./app.py)
   ```shell
   python app.py
   ```
