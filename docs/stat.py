#!/usr/bin/env python
import functools as func
import glob
import re
import numpy as np

import titlecase

files = sorted(glob.glob('*_models.md'))

stats = []

for f in files:
    with open(f, 'r') as content_file:
        content = content_file.read()

    # title
    title = content.split('\n')[0].replace('#', '')

    # count papers
    papers = set(
        (papertype, titlecase.titlecase(paper.lower().strip()))
        for (papertype, paper) in re.findall(r'\[([A-Z]*?)\].*?\btitle\s*=\s*{(.*?)}', content, re.DOTALL))
    paperlist = '\n'.join(sorted(f'    - [{t}] {x}' for t, x in papers))
    # count configs
    configs = set(x.lower().strip()
                  for x in re.findall(r'https.*configs/.*\.py', content))

    # count ckpts
    ckpts = set(x.lower().strip()
                for x in re.findall(r'https://download.*\.pth', content)
                if 'mmpose' in x)

    statsmsg = f"""
## [{title}]({f})

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
papertypes, papercounts = np.unique([t for t, _ in allpapers], return_counts=True)
countstr = '\n'.join([f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# Model Zoo Statistics

* Number of checkpoints: {len(allckpts)}
* Number of configs: {len(allconfigs)}
* Number of papers: {len(allpapers)}
{countstr}

{msglist}

"""

with open('modelzoo.md', 'w') as f:
    f.write(modelzoo)
