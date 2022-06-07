#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
from glob import glob

from titlecase import titlecase

os.makedirs('topics', exist_ok=True)
os.makedirs('papers', exist_ok=True)


def _parse_task(task):
    """Parse task name.

    Data modality is represented by a string of 4 or 5 parts like:
    - 2d_kpt_sview_rgb_img
    - gesture_sview_rgbd_vid
    """

    parts = task.split('_')
    if len(parts) == 5:
        pass
    elif len(parts) == 4:
        # The first part "spatial dimension" is optional
        parts = [''] + parts
    else:
        raise ValueError('Invalid modality')

    return parts


# Step 1: get subtopics: a mix of topic and task
minisections = [
    x.split(osp.sep)[-2:] for x in glob('../../configs/*/*')
    if '_base_' not in x
]
alltopics = sorted(list(set(x[0] for x in minisections)))
subtopics = []
for topic in alltopics:
    tasks = [_parse_task(x[1]) for x in minisections if x[0] == topic]
    valid_ids = []
    for i in range(len(tasks[0])):
        if len(set(x[i] for x in tasks)) > 1:
            valid_ids.append(i)
    if len(valid_ids) > 0:
        for task in tasks:
            appendix = ','.join(
                [task[i].title() for i in valid_ids if task[i]])
            subtopic = [
                f'{titlecase(topic)}({appendix})',
                topic,
                '_'.join(t for t in task if t),
            ]
            subtopics.append(subtopic)
    else:
        subtopics.append([titlecase(topic), topic, '_'.join(tasks[0])])

contents = {}
for subtopic, topic, task in sorted(subtopics):
    # Step 2: get all datasets
    datasets = sorted(
        list(
            set(
                x.split(osp.sep)[-2]
                for x in glob(f'../../configs/{topic}/{task}/*/*/'))))
    contents[subtopic] = {d: {} for d in datasets}
    for dataset in datasets:
        # Step 3: get all settings: algorithm + backbone + trick
        for file in glob(f'../../configs/{topic}/{task}/*/{dataset}/*.md'):
            keywords = (file.split(osp.sep)[-3],
                        *file.split(osp.sep)[-1].split('_')[:-1])
            with open(file, 'r', encoding='utf-8') as f:
                contents[subtopic][dataset][keywords] = f.read()

# Step 4: write files by topic
for subtopic, datasets in contents.items():
    lines = [f'# {subtopic}', '']
    for dataset, keywords in datasets.items():
        if len(keywords) == 0:
            continue
        lines += [
            '<hr/>', '<br/><br/>', '', f'## {titlecase(dataset)} Dataset', ''
        ]
        for keyword, info in keywords.items():
            keyword_strs = [titlecase(x.replace('_', ' ')) for x in keyword]
            lines += [
                '<br/>', '',
                (f'### {" + ".join(keyword_strs)}'
                 f' on {titlecase(dataset)}'), '', info, ''
            ]

    with open(f'topics/{subtopic.lower()}.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# Step 5: write files by paper
allfiles = [x.split(osp.sep)[-2:] for x in glob('../en/papers/*/*.md')]
sections = sorted(list(set(x[0] for x in allfiles)))
for section in sections:
    lines = [f'# {titlecase(section)}', '']
    files = [f for s, f in allfiles if s == section]
    for file in files:
        with open(
                f'../en/papers/{section}/{file}', 'r', encoding='utf-8') as f:
            keyline = [
                line for line in f.readlines() if line.startswith('<summary')
            ][0]
        papername = re.sub(r'\<.*?\>', '', keyline).strip()
        paperlines = []
        for subtopic, datasets in contents.items():
            for dataset, keywords in datasets.items():
                keywords = {k: v for k, v in keywords.items() if keyline in v}
                if len(keywords) == 0:
                    continue
                for keyword, info in keywords.items():
                    keyword_strs = [
                        titlecase(x.replace('_', ' ')) for x in keyword
                    ]
                    paperlines += [
                        '<br/>', '',
                        (f'### {" + ".join(keyword_strs)}'
                         f' on {titlecase(dataset)}'), '', info, ''
                    ]
        if len(paperlines) > 0:
            lines += ['<hr/>', '<br/><br/>', '', f'## {papername}', '']
            lines += paperlines

    with open(f'papers/{section}.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
