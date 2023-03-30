#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
from glob import glob


def _get_project_docs():
    """Get all project document files.

    Returns:
        list[str]: file paths
    """
    project_root = osp.join('..', '..', 'projects')
    pattern = osp.sep.join(['*'] * 2) + '.md'
    docs = glob(osp.join(project_root, pattern))
    docs = [
        doc for doc in docs
        if 'example_project' not in doc and '_CN' not in doc
    ]
    return docs


def _parse_project_doc_path(fn):
    """Get project name and banner from a project reference file.

    Returns:
        tuple:
        - project_name (str)
        - project_banner (str)
    """
    project_banner, project_name = None, None
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if re.match('^( )*<img', line) and not project_banner:
                project_banner = line
            if line.startswith('# ') and not project_name:
                project_name = line
            if project_name and project_banner:
                break
    if project_name is None or project_banner is None:
        raise ValueError(f'Invalid paper reference file {fn}')

    project_name = re.sub(r'^\# ', '', project_name).strip()
    project_banner = project_banner.strip()
    return project_name, project_banner


def _get_project_intro_doc():
    project_intro_doc = []
    with open(
            osp.join('..', '..', 'projects', 'README.md'), 'r',
            encoding='utf-8') as f:
        for line in f.readlines():
            if line.startswith('# Welcome'):
                continue
            if './faq.md' in line:
                line = line.replace('./faq.md', '#faq')
            if 'example_project' in line:
                line = line.replace(
                    './', 'https://github.com/open-mmlab/mmpose/'
                    'tree/dev-1.x/projects/')
            project_intro_doc.append(line)
            if line.startswith('## Project List'):
                break
    return project_intro_doc


def _get_faq_doc():
    faq_doc = []
    with open(
            osp.join('..', '..', 'projects', 'faq.md'), 'r',
            encoding='utf-8') as f:
        for line in f.readlines():
            if '#' in line:
                line = re.sub(r'^(\#+)', r'\g<1>#', line)
            faq_doc.append(line)
    return faq_doc


def main():

    # Build output folders
    os.makedirs('projects', exist_ok=True)

    # Collect all document contents
    project_doc_list = _get_project_docs()

    project_lines = []
    for path in project_doc_list:
        name, banner = _parse_project_doc_path(path)
        _path = path.split(osp.sep)
        _rel_path = _path[_path.index('projects'):-1]
        url = 'https://github.com/open-mmlab/mmpose/blob/dev-1.x/' + '/'.join(
            _rel_path)
        _name = name.split(':', 1)
        name, description = _name[0], '' if len(
            _name) < 2 else f': {_name[-1]}'
        project_lines += [
            f'- **{name}**{description} [\\[github\\]]({url})', '',
            '<div align="center">', ' ' + banner, '</div>', '<br/>', ''
        ]

    project_intro_doc = _get_project_intro_doc()
    faq_doc = _get_faq_doc()

    with open(
            osp.join('projects', 'community_projects.md'), 'w',
            encoding='utf-8') as f:
        f.write('# Projects of MMPose from Community Contributors\n')
        f.write(''.join(project_intro_doc))
        f.write('\n'.join(project_lines))
        f.write(''.join(faq_doc))


if __name__ == '__main__':
    print('collect project documents')
    main()
