# Just Dance - A Simple Implementation

This project presents a dance scoring system based on RTMPose. Users can compare the similarity between two dancers in different videos: one referred to as the "teacher video" and the other as the "student video."

Here is an example of the output dance comparison:

![output](https://github.com/open-mmlab/mmpose/assets/26127467/56d5c4d1-55d8-4222-b481-2418cc29a8d4)

## Usage

### Jupyter Notebook

We provide a Jupyter Notebook [`just_dance_demo.ipynb`](./just_dance_demo.ipynb) that contains the complete process of dance comparison. It includes steps such as video FPS adjustment, pose estimation, snippet alignment, scoring, and the generation of the merged video.

### CLI tool

Users can simply run the following command to generate the comparison video:

```shell
python process_video ${TEACHER_VIDEO} ${STUDENT_VIDEO}
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
