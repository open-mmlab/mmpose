#!/usr/bin/env python
import glob
import os
import re

import titlecase


def anchor(name):
    return re.sub(r'-+', '-', re.sub(r'[^a-zA-Z0-9]', '-',
                                     name.strip().lower()))


# Count algorithms
root_dir = '../configs'

for root, dirs, files in os.walk(root_dir, topdown=True):
    if '_result' in dirs and '_result_collection' in dirs:
        collect_files = sorted(
            glob.glob(os.path.join(root, '_result_collection/*.md')))

        for cf in collect_files:
            with open(cf, 'r') as collect_content_file:
                collect_content = collect_content_file.read()
                collect_papers = set((
                    papertype, titlecase.titlecase(paper.lower().strip())
                ) for (papertype, paper) in re.findall(
                    r'<!--\s*\[([A-Z]*?)\]\s*-->\s*\n.*?\btitle\s*=\s*{(.*?)}',
                    collect_content, re.DOTALL))

            res_files = sorted(glob.glob(os.path.join(root, '_result/*.md')))

            for rf in res_files:
                with open(rf, 'r') as res_content_file:
                    res_content = res_content_file.read()
                    res_papers = set(
                        (papertype, titlecase.titlecase(paper.lower().strip()))
                        for (papertype, paper) in re.findall(
                            r'<!--\s*\[([A-Z]*?)\]\s*-->'
                            r'\s*\n.*?\btitle\s*=\s*{(.*?)}', res_content,
                            re.DOTALL))

                    Flag = True
                    for paper in collect_papers:
                        if paper not in res_papers:
                            Flag = False
                            break
                    if Flag:
                        with open(cf, 'a') as file_out:
                            file_out.write('\n' + res_content.split('# ')[-1])
