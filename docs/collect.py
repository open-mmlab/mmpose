#!/usr/bin/env python
import glob
import os

from titlecase import titlecase

os.makedirs('topics', exist_ok=True)

# Step 1: get subtopics: a mix of topic and task
sections = [x.split('/')[-2:] for x in glob.glob('../configs/*/*')]
alltopics = sorted(list(set(x[0] for x in sections)))
subtopics = []
for t in alltopics:
    data = [x[1].split('_') for x in sections if x[0] == t]
    valid_ids = []
    for i in range(len(data[0])):
        if len(set(x[i] for x in data)) > 1:
            valid_ids.append(i)
    if len(valid_ids) > 0:
        subtopics.extend(
            [f"{t}({','.join([d[i] for i in valid_ids])})", t, '_'.join(d)]
            for d in data)
    else:
        subtopics.append([t, t, '_'.join(data[0])])

contents = {}
for subtopic, topic, task in sorted(subtopics):
    # Step 2: get all datasets
    datasets = sorted(
        list(
            set(
                x.split('/')[-2]
                for x in glob.glob(f'../configs/{topic}/{task}/*/*/'))))
    contents[subtopic] = {d: {} for d in datasets}
    for dataset in datasets:
        # Step 3: get all settings: algorithm + backbone + trick
        for file in glob.glob(f'../configs/{topic}/{task}/*/{dataset}/*.md'):
            keywords = (file.split('/')[-3],
                        *file.split('/')[-1].split('_')[:-1])
            with open(file, 'r') as f:
                contents[subtopic][dataset][keywords] = f.read()

# Step 4: write files
for subtopic, datasets in contents.items():
    lines = [f'# {titlecase(subtopic)}', '']
    for dataset, keywords in datasets.items():
        if len(keywords) == 0:
            continue
        lines += ['<hr/>', '', f'## {titlecase(dataset)}', '']
        for keyword, info in keywords.items():
            lines += [
                f'### {" + ".join([titlecase(x.replace("_", " ")) for x in keyword])}',
                '', info, ''
            ]

    with open(f'topics/{subtopic}.md', 'w') as f:
        f.write('\n'.join(lines))
