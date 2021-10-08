# Copyright (c) OpenMMLab. All rights reserved.
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import sys

import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'MMPose'
copyright = '2020-2021, OpenMMLab'
author = 'MMPose Authors'

# The full version, including alpha/beta/rc tags
version_file = '../mmpose/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
    'sphinx_markdown_tables', 'sphinx_copybutton', 'myst_parser'
]

autodoc_mock_imports = ['json_tricks', 'mmpose.version']

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    'menu': [
        {
            'name':
            'Tutorial',
            'url':
            'https://colab.research.google.com/github/'
            'open-mmlab/mmpose/blob/master/demo/MMPose_Tutorial.ipynb'
        },
        {
            'name': 'GitHub',
            'url': 'https://github.com/open-mmlab/mmpose'
        },
        {
            'name':
            'Projects',
            'children': [
                {
                    'name': 'MMCV',
                    'url': 'https://github.com/open-mmlab/mmcv',
                    'description': '计算机视觉基础库'
                },
                {
                    'name': 'MMDetection',
                    'url': 'https://github.com/open-mmlab/mmdetection',
                    'description': '检测工具箱与测试基准'
                },
                {
                    'name': 'MMAction2',
                    'url': 'https://github.com/open-mmlab/mmaction2',
                    'description': '视频理解工具箱与测试基准'
                },
                {
                    'name': 'MMClassification',
                    'url': 'https://github.com/open-mmlab/mmclassification',
                    'description': '图像分类工具箱与测试基准'
                },
                {
                    'name': 'MMSegmentation',
                    'url': 'https://github.com/open-mmlab/mmsegmentation',
                    'description': '语义分割工具箱与测试基准'
                },
                {
                    'name': 'MMDetection3D',
                    'url': 'https://github.com/open-mmlab/mmdetection3d',
                    'description': '通用3D目标检测平台'
                },
                {
                    'name': 'MMEditing',
                    'url': 'https://github.com/open-mmlab/mmediting',
                    'description': '图像视频编辑工具箱'
                },
                {
                    'name': 'MMOCR',
                    'url': 'https://github.com/open-mmlab/mmocr',
                    'description': '全流程文字检测识别理解工具包'
                },
                {
                    'name': 'MMTracking',
                    'url': 'https://github.com/open-mmlab/mmtracking',
                    'description': '一体化视频目标感知平台'
                },
                {
                    'name': 'MMGeneration',
                    'url': 'https://github.com/open-mmlab/mmgeneration',
                    'description': '生成模型工具箱'
                },
            ]
        },
        {
            'name':
            'OpenMMLab',
            'children': [
                {
                    'name': '主页',
                    'url': 'https://openmmlab.com/'
                },
                {
                    'name': 'GitHub',
                    'url': 'https://github.com/open-mmlab/'
                },
            ]
        },
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

language = 'zh_CN'

html_static_path = ['_static']
html_css_files = ['css/readthedocs.css']

# Enable ::: for my_st
myst_enable_extensions = ['colon_fence']

master_doc = 'index'


def builder_inited_handler(app):
    subprocess.run(['./collect.py'])
    subprocess.run(['./merge_docs.sh'])
    subprocess.run(['./stats.py'])


def setup(app):
    app.connect('builder-inited', builder_inited_handler)
