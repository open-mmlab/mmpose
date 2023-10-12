# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from functools import partial
from typing import Optional

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
mmpose_path = project_path.split('/projects', 1)[0]

os.system('python -m pip install Openmim')
os.system('python -m mim install "mmcv>=2.0.0"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system(f'python -m mim install -e {mmpose_path}')

os.environ['PATH'] = f"{os.environ['PATH']}:{project_path}"
os.environ[
    'PYTHONPATH'] = f"{os.environ.get('PYTHONPATH', '.')}:{project_path}"
sys.path.append(project_path)

import gradio as gr  # noqa
from mmengine.utils import mkdir_or_exist  # noqa
from process_video import VideoProcessor  # noqa


def process_video(
    teacher_video: Optional[str] = None,
    student_video: Optional[str] = None,
):
    print(teacher_video)
    print(student_video)

    video_processor = VideoProcessor()
    if student_video is None and teacher_video is not None:
        # Pre-process the teacher video when users record the student video
        # using a webcam. This allows users to view the teacher video and
        # follow the dance moves while recording the student video.
        _ = video_processor.get_keypoints_from_video(teacher_video)
        return teacher_video
    elif teacher_video is None and student_video is not None:
        _ = video_processor.get_keypoints_from_video(student_video)
        return student_video
    elif teacher_video is None and student_video is None:
        return None

    return video_processor.run(teacher_video, student_video)


# download video resources
mkdir_or_exist(os.path.join(project_path, 'resources'))
os.system(
    f'wget -O {project_path}/resources/tom.mp4 https://download.openmmlab.com/mmpose/v1/projects/just_dance/tom.mp4'  # noqa
)
os.system(
    f'wget -O {project_path}/resources/idol_producer.mp4 https://download.openmmlab.com/mmpose/v1/projects/just_dance/idol_producer.mp4'  # noqa
)
os.system(
    f'wget -O {project_path}/resources/tsinghua_30fps.mp4 https://download.openmmlab.com/mmpose/v1/projects/just_dance/tsinghua_30fps.mp4'  # noqa
)
os.system(
    f'wget -O {project_path}/resources/student1.mp4 https://download.openmmlab.com/mmpose/v1/projects/just_dance/student1.mp4'  # noqa
)
os.system(
    f'wget -O {project_path}/resources/bear_teacher.mp4 https://download.openmmlab.com/mmpose/v1/projects/just_dance/bear_teacher.mp4'  # noqa
)

with gr.Blocks() as demo:
    with gr.Tab('Upload-Video'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('Student Video')
                student_video = gr.Video(type='mp4')
                gr.Examples([
                    os.path.join(project_path, 'resources/tom.mp4'),
                    os.path.join(project_path, 'resources/tsinghua_30fps.mp4'),
                    os.path.join(project_path, 'resources/student1.mp4')
                ], student_video)
            with gr.Column():
                gr.Markdown('Teacher Video')
                teacher_video = gr.Video(type='mp4')
                gr.Examples([
                    os.path.join(project_path, 'resources/idol_producer.mp4'),
                    os.path.join(project_path, 'resources/bear_teacher.mp4')
                ], teacher_video)

        button = gr.Button('Grading', variant='primary')
        gr.Markdown('## Display')
        out_video = gr.Video()

        button.click(
            partial(process_video), [teacher_video, student_video], out_video)

    with gr.Tab('Webcam-Video'):
        with gr.Row():
            with gr.Column():
                gr.Markdown('Student Video')
                student_video = gr.Video(source='webcam', type='mp4')
            with gr.Column():
                gr.Markdown('Teacher Video')
                teacher_video = gr.Video(type='mp4')
                gr.Examples([
                    os.path.join(project_path, 'resources/idol_producer.mp4')
                ], teacher_video)
                button_upload = gr.Button('Upload', variant='primary')

        button = gr.Button('Grading', variant='primary')
        gr.Markdown('## Display')
        out_video = gr.Video()

        button_upload.click(
            partial(process_video), [teacher_video, student_video], out_video)
        button.click(
            partial(process_video), [teacher_video, student_video], out_video)

gr.close_all()
demo.queue()
demo.launch()
