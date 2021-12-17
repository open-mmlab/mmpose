#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os
import re
from glob import glob

from titlecase import titlecase

os.makedirs('topics', exist_ok=True)
os.makedirs('papers', exist_ok=True)

# Step 1: get subtopics: a mix of topic and task
minisections = [
    x.split('/')[-2:] for x in glob('../../configs/*/*') if '_base_' not in x
]
alltopics = sorted(list(set(x[0] for x in minisections)))
subtopics = []
for t in alltopics:
    data = [x[1].split('_') for x in minisections if x[0] == t]
    valid_ids = []
    for i in range(len(data[0])):
        if len(set(x[i] for x in data)) > 1:
            valid_ids.append(i)
    if len(valid_ids) > 0:
        subtopics.extend([
            f"{titlecase(t)}({','.join([d[i].title() for i in valid_ids])})",
            t, '_'.join(d)
        ] for d in data)
    else:
        subtopics.append([titlecase(t), t, '_'.join(data[0])])

contents = {}
for subtopic, topic, task in sorted(subtopics):
    # Step 2: get all datasets
    datasets = sorted(
        list(
            set(
                x.split('/')[-2]
                for x in glob(f'../../configs/{topic}/{task}/*/*/'))))
    contents[subtopic] = {d: {} for d in datasets}
    for dataset in datasets:
        # Step 3: get all settings: algorithm + backbone + trick
        for file in glob(f'../../configs/{topic}/{task}/*/{dataset}/*.md'):
            keywords = (file.split('/')[-3],
                        *file.split('/')[-1].split('_')[:-1])
            with open(file, 'r') as f:
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

    with open(f'topics/{subtopic.lower()}.md', 'w') as f:
        f.write('\n'.join(lines))

# Step 5: write files by paper
allfiles = [x.split('/')[-2:] for x in glob('../en/papers/*/*.md')]
sections = sorted(list(set(x[0] for x in allfiles)))
for section in sections:
    lines = [f'# {titlecase(section)}', '']
    files = [f for s, f in allfiles if s == section]
    for file in files:
        with open(f'../en/papers/{section}/{file}', 'r') as f:
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

    with open(f'papers/{section}.md', 'w') as f:
        f.write('\n'.join(lines))
