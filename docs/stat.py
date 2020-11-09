#!/usr/bin/env python
import re
import glob
import functools as func

files = sorted(glob.glob('*_models.md'))

stats = []

for f in files:
    with open(f, 'r') as content_file:
        content = content_file.read()

    # title
    title = content.split('\n')[0]

    # count papers
    papers = set(x.lower().strip() for x in re.findall(r'\btitle={(.*)}', content))
    paperlist = '\n'.join('* ' + x for x in papers)
    if len(papers) > 0:
        paperlist = '### Supported Papers\n\n' + paperlist

    # count configs
    configs = set(x.lower().strip() for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip() for x in re.findall(r'https://download.*\.pth', content))

    statsmsg = f"""
#{title}

* Number of checkpoints: {len(ckpts)}
* Number of configs: {len(configs)}
* Number of papers: {len(papers)}

{paperlist}

    """

    stats.append((papers, configs, ckpts, statsmsg))

allpapers = func.reduce(lambda a, b: a.union(b), [p for p, _, _, _ in stats])
allconfigs = func.reduce(lambda a, b: a.union(b), [c for _, c, _, _ in stats])
allckpts = func.reduce(lambda a, b: a.union(b), [c for _, _, c, _ in stats])
msglist = '\n'.join(x for _, _, _, x in stats)

modelzoo = f"""
# Model Zoo Statistics

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}

{msglist}

"""

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)
