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

collect_files = sorted(glob.glob('../configs/README/*.md'))
for cf in collect_files:
    res_dict = {}
    with open(cf, 'r') as collect_content_file:
        collect_content = collect_content_file.read()
        collect_papers = set(
            (papertype, titlecase.titlecase(paper.lower().strip()))
            for (papertype, paper) in re.findall(
                r'<!--\s*\[([A-Z]*?)\]\s*-->\s*\n.*?\btitle\s*=\s*{(.*?)}',
                collect_content, re.DOTALL))

    for root, dirs, files in os.walk(root_dir, topdown=True):
        SPL = root.split('/')
        if '_result' in dirs:
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
                        topic = SPL[-3]
                        task = SPL[-2]
                        algorithm = SPL[-1]

                        if task == '2d_kpt_sview_rgb_img':
                            task_str = 'Single-view RGB image based 2D ' \
                                f'{topic} keypoint estimation'
                        elif task == '3d_kpt_sview_rgb_img':
                            task_str = 'Single-view RGB image based 3D ' \
                                f'{topic} keypoint estimation'
                        elif task == '3d_kpt_sview_rgb_vid':
                            task_str = 'Single-view RGB video based 3D ' \
                                f'{topic} keypoint estimation'
                        elif task == '3d_mesh_sview_rgb_img':
                            task_str = 'Single-view RGB image based 3D ' \
                                f'{topic} mesh recovery'

                        if task_str not in res_dict:
                            res_dict[task_str] = [
                                res_content[res_content.find('####'):]
                            ]
                        else:
                            res_dict[task_str].append(
                                res_content[res_content.find('####'):])

    with open(cf, 'a') as file_out:
        for (key, value) in res_dict.items():
            file_out.write(f'\n### {key}\n\n')
            for v in value:
                file_out.write(v)
                file_out.write('\n')
