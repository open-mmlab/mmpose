# Copyright (c) OpenMMLab. All rights reserved.

import os
from functools import partial

# prepare environment
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
mmpose_path = project_path.split('/projects', 1)[0]

os.system('python -m pip install Openmim')
os.system('python -m pip install openxlab')
os.system('python -m pip install gradio==3.38.0')

os.system('python -m mim install "mmcv>=2.0.0"')
os.system('python -m mim install "mmengine>=0.9.0"')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system(f'python -m mim install -e {mmpose_path}')

import gradio as gr  # noqa
from openxlab.model import download  # noqa

from mmpose.apis import MMPoseInferencer  # noqa

# download checkpoints
download(model_repo='mmpose/RTMPose', model_name='dwpose-l')
download(model_repo='mmpose/RTMPose', model_name='RTMW-x')
download(model_repo='mmpose/RTMPose', model_name='RTMO-l')
download(model_repo='mmpose/RTMPose', model_name='RTMPose-l-body8')
download(model_repo='mmpose/RTMPose', model_name='RTMPose-m-face6')

models = [
    'rtmpose | body', 'rtmo | body', 'rtmpose | face', 'dwpose | wholebody',
    'rtmw | wholebody'
]
cached_model = {model: None for model in models}


def predict(input,
            draw_heatmap=False,
            model_type='body',
            skeleton_style='mmpose',
            input_type='image'):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    if model_type == 'rtmpose | face':
        if cached_model[model_type] is None:
            cached_model[model_type] = MMPoseInferencer(pose2d='face')
        model = cached_model[model_type]

    elif model_type == 'dwpose | wholebody':
        if cached_model[model_type] is None:
            cached_model[model_type] = MMPoseInferencer(
                pose2d=os.path.join(
                    project_path, 'rtmpose/wholebody_2d_keypoint/'
                    'rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'),
                pose2d_weights='https://download.openmmlab.com/mmpose/v1/'
                'projects/rtmposev1/rtmpose-l_simcc-ucoco_dw-ucoco_270e-'
                '384x288-2438fd99_20230728.pth')
        model = cached_model[model_type]

    elif model_type == 'rtmw | wholebody':
        if cached_model[model_type] is None:
            cached_model[model_type] = MMPoseInferencer(
                pose2d=os.path.join(
                    project_path, 'rtmpose/wholebody_2d_keypoint/'
                    'rtmw-l_8xb320-270e_cocktail14-384x288.py'),
                pose2d_weights='https://download.openmmlab.com/mmpose/v1/'
                'projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-'
                '384x288-20231122.pth')
        model = cached_model[model_type]

    elif model_type == 'rtmpose | body':
        if cached_model[model_type] is None:
            cached_model[model_type] = MMPoseInferencer(pose2d='rtmpose-l')
        model = cached_model[model_type]

    elif model_type == 'rtmo | body':
        if cached_model[model_type] is None:
            cached_model[model_type] = MMPoseInferencer(pose2d='rtmo')
        model = cached_model[model_type]
        draw_heatmap = False

    else:
        raise ValueError

    if input_type == 'image':

        result = next(
            model(
                input,
                return_vis=True,
                draw_heatmap=draw_heatmap,
                skeleton_style=skeleton_style))
        img = result['visualization'][0][..., ::-1]
        return img

    elif input_type == 'video':

        for _ in model(
                input,
                vis_out_dir='test.mp4',
                draw_heatmap=draw_heatmap,
                skeleton_style=skeleton_style):
            pass

        return 'test.mp4'

    return None


news_list = [
    '2023-8-1: We support [DWPose](https://arxiv.org/pdf/2307.15880.pdf).',
    '2023-9-25: We release an alpha version of RTMW model, the technical '
    'report will be released soon.',
    '2023-12-11: Update RTMW models, the online version is the RTMW-l with '
    '70.1 mAP on COCO-Wholebody.',
    '2023-12-13: We release an alpha version of RTMO (One-stage RTMPose) '
    'models.',
]

with gr.Blocks() as demo:

    with gr.Tab('Upload-Image'):
        input_img = gr.Image(type='numpy')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(models, label='Model | Keypoint Type')

        gr.Markdown('## News')
        for news in news_list[::-1]:
            gr.Markdown(news)

        gr.Markdown('## Output')
        out_image = gr.Image(type='numpy')
        gr.Examples(['./tests/data/coco/000000000785.jpg'], input_img)
        input_type = 'image'
        button.click(
            partial(predict, input_type=input_type),
            [input_img, hm, model_type], out_image)

    with gr.Tab('Webcam-Image'):
        input_img = gr.Image(source='webcam', type='numpy')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(models, label='Model | Keypoint Type')

        gr.Markdown('## News')
        for news in news_list[::-1]:
            gr.Markdown(news)

        gr.Markdown('## Output')
        out_image = gr.Image(type='numpy')

        input_type = 'image'
        button.click(
            partial(predict, input_type=input_type),
            [input_img, hm, model_type], out_image)

    with gr.Tab('Upload-Video'):
        input_video = gr.Video(type='mp4')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(models, label='Model | Keypoint type')

        gr.Markdown('## News')
        for news in news_list[::-1]:
            gr.Markdown(news)

        gr.Markdown('## Output')
        out_video = gr.Video()

        input_type = 'video'
        button.click(
            partial(predict, input_type=input_type),
            [input_video, hm, model_type], out_video)

    with gr.Tab('Webcam-Video'):
        input_video = gr.Video(source='webcam', format='mp4')
        button = gr.Button('Inference', variant='primary')
        hm = gr.Checkbox(label='draw-heatmap', info='Whether to draw heatmap')
        model_type = gr.Dropdown(models, label='Model | Keypoint Type')

        gr.Markdown('## News')
        for news in news_list[::-1]:
            gr.Markdown(news)

        gr.Markdown('## Output')
        out_video = gr.Video()

        input_type = 'video'
        button.click(
            partial(predict, input_type=input_type),
            [input_video, hm, model_type], out_video)

gr.close_all()
demo.queue()
demo.launch()
