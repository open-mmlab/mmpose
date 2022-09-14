#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
from collections import defaultdict
from glob import glob

from addict import Addict
from titlecase import titlecase


def _get_model_docs():
    """Get all model document files.

    Returns:
        list[str]: file paths
    """
    config_root = osp.join('..', '..', 'configs')
    pattern = osp.sep.join(['*'] * 4) + '.md'
    docs = glob(osp.join(config_root, pattern))
    docs = [doc for doc in docs if '_base_' not in doc]
    return docs


def _parse_model_doc_path(path):
    """Parse doc file path.

    Typical path would be like:

        configs/<task>/<algorithm>/<dataset>/<setting>.md

    An example is:

        "configs/animal_2d_keypoint/topdown_heatmap/
        animalpose/resnet_animalpose.md"

    Returns:
        tuple:
        - task (str): e.g. ``'Animal 2D Keypoint'``
        - dataset (str): e.g. ``'animalpose'``
        - keywords (tuple): e.g. ``('topdown heatmap', 'resnet')``
    """
    _path = path.split(osp.sep)
    _rel_path = _path[_path.index('configs'):]

    # get task
    def _titlecase_callback(word, **kwargs):
        if word == '2d':
            return '2D'
        if word == '3d':
            return '3D'

    task = titlecase(
        _rel_path[1].replace('_', ' '), callback=_titlecase_callback)

    # get dataset
    dataset = _rel_path[3]

    # get keywords
    keywords_algo = (_rel_path[2], )
    keywords_setting = tuple(_rel_path[4][:-3].split('_'))
    keywords = keywords_algo + keywords_setting

    return task, dataset, keywords


def _get_paper_refs():
    """Get all paper references.

    Returns:
        Dict[str, List[str]]: keys are paper categories and values are lists
        of paper paths.
    """
    papers = glob('../src/papers/*/*.md')
    paper_refs = defaultdict(list)
    for fn in papers:
        category = fn.split(osp.sep)[3]
        paper_refs[category].append(fn)

    return paper_refs


def _parse_paper_ref(fn):
    """Get paper name and indicator pattern from a paper reference file.

    Returns:
        tuple:
        - paper_name (str)
        - paper_indicator (str)
    """
    indicator = None
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.startswith('<summary'):
                indicator = line
                break
    if indicator is None:
        raise ValueError(f'Invalid paper reference file {fn}')

    paper_name = re.sub(r'\<.*?\>', '', indicator).strip()
    return paper_name, indicator


def main():

    # Build output folders
    os.makedirs('model_zoo', exist_ok=True)
    os.makedirs('model_zoo_papers', exist_ok=True)

    # Collect all document contents
    model_doc_list = _get_model_docs()
    model_docs = Addict()

    for path in model_doc_list:
        task, dataset, keywords = _parse_model_doc_path(path)
        with open(path, 'r', encoding='utf-8') as f:
            doc = {
                'task': task,
                'dataset': dataset,
                'keywords': keywords,
                'path': path,
                'content': f.read()
            }
        model_docs[task][dataset][keywords] = doc

    # Write files by task
    for task, dataset_dict in model_docs.items():
        lines = [f'# {task}', '']
        for dataset, keywords_dict in dataset_dict.items():
            lines += [
                '<hr/>', '<br/><br/>', '', f'## {titlecase(dataset)} Dataset',
                ''
            ]

            for keywords, doc in keywords_dict.items():
                keyword_strs = [
                    titlecase(x.replace('_', ' ')) for x in keywords
                ]
                dataset_str = titlecase(dataset)
                if dataset_str in keyword_strs:
                    keyword_strs.remove(dataset_str)

                lines += [
                    '<br/>', '',
                    (f'### {" + ".join(keyword_strs)}'
                     f' on {dataset_str}'), '', doc['content'], ''
                ]

        fn = osp.join('model_zoo', f'{task.replace(" ", "_").lower()}.md')
        with open(fn, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    # Write files by paper
    paper_refs = _get_paper_refs()

    for paper_cat, paper_list in paper_refs.items():
        lines = []
        for paper_fn in paper_list:
            paper_name, indicator = _parse_paper_ref(paper_fn)
            paperlines = []
            for task, dataset_dict in model_docs.items():
                for dataset, keywords_dict in dataset_dict.items():
                    for keywords, doc_info in keywords_dict.items():

                        if indicator not in doc_info['content']:
                            continue

                        keyword_strs = [
                            titlecase(x.replace('_', ' ')) for x in keywords
                        ]

                        dataset_str = titlecase(dataset)
                        if dataset_str in keyword_strs:
                            keyword_strs.remove(dataset_str)
                        paperlines += [
                            '<br/>', '',
                            (f'### {" + ".join(keyword_strs)}'
                             f' on {dataset_str}'), '', doc_info['content'], ''
                        ]
            if paperlines:
                lines += ['<hr/>', '<br/><br/>', '', f'## {paper_name}', '']
                lines += paperlines

        if lines:
            lines = [f'# {titlecase(paper_cat)}', ''] + lines
            with open(
                    osp.join('model_zoo_papers', f'{paper_cat.lower()}.md'),
                    'w',
                    encoding='utf-8') as f:
                f.write('\n'.join(lines))


if __name__ == '__main__':
    print('collect model zoo documents')
    main()
