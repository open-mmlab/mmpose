#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

import glob
import os.path as osp
import re
import sys

import mmcv

MMPOSE_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))


def dump_yaml_and_check_difference(obj, file):
    """Dump object to a yaml file, and check if the file content is different
    from the original.

    Args:
        obj (any): The python object to be dumped.
        file (str): YAML filename to dump the object to.
    Returns:
        Bool: If the target YAML file is different from the original.
    """

    str_dump = mmcv.dump(obj, None, file_format='yaml', sort_keys=True)

    if osp.isfile(file):
        file_exists = True
        with open(file, 'r', encoding='utf-8') as f:
            str_orig = f.read()
    else:
        file_exists = False
        str_orig = None

    if file_exists and str_orig == str_dump:
        is_different = False
    else:
        is_different = True
        with open(file, 'w', encoding='utf-8') as f:
            f.write(str_dump)

    return is_different


def collate_metrics(keys):
    """Collect metrics from the first row of the table.

    Args:
        keys (List): Elements in the first row of the table.

    Returns:
        List: A list of metrics.
    """
    all_metrics = [
        'acc', 'ap', 'ar', 'pck', 'auc', '3dpck', 'p-3dpck', '3dauc',
        'p-3dauc', 'epe', 'nme', 'mpjpe', 'p-mpjpe', 'n-mpjpe', 'mean', 'head',
        'sho', 'elb', 'wri', 'hip', 'knee', 'ank', 'total'
    ]
    used_metrics = []
    metric_idx = []
    for idx, key in enumerate(keys):
        if key in ['Arch', 'Input Size', 'ckpt', 'log']:
            continue
        for metric in all_metrics:
            if metric.upper() in key or metric.capitalize() in key:
                used_metric = ''
                i = 0
                while i < len(key):
                    # skip ``<...>``
                    if key[i] == '<':
                        while key[i] != '>':
                            i += 1
                    # omit bold or italic
                    elif key[i] == '*' or key[i] == '_':
                        used_metric += ' '
                    else:
                        used_metric += key[i]
                    i += 1
                re.sub(' +', ' ', used_metric)
                used_metric = used_metric.strip()
                if metric in ['ap', 'ar']:
                    match = re.search(r'\d+', used_metric)
                    if match is not None:
                        l, r = match.span(0)
                        digits = match.group(0)
                        used_metric = used_metric[:l] + '@' + \
                            str(int(digits) * 0.01) + used_metric[r:]
                used_metrics.append(used_metric)
                metric_idx.append(idx)
                break
    return used_metrics, metric_idx


def collect_paper_readme():
    """Collect paper readme files for collections.

    Returns:
        dict: collection name to corresponding paper readme link.
    """
    link_prefix = 'https://github.com/open-mmlab/mmpose/blob/master/'

    readme_files = glob.glob(osp.join('docs/en/papers/*/*.md'))
    readme_files.sort()
    collection2readme = {}

    for readme_file in readme_files:
        with open(readme_file, encoding='utf-8') as f:
            keyline = [
                line for line in f.readlines() if line.startswith('<summary')
            ][0]
            name = re.findall(r'<a href=".*">(.*?)[ ]*\(.*\'.*\).*</a>',
                              keyline)[0]
            collection2readme[name] = link_prefix + readme_file.replace(
                '\\', '/')

    return collection2readme


def parse_config_path(path):
    """Parse model information from the config path.

    Args:
        path (str): a path under the configs folder

    Returns:
        dict: model information with following fields
            - target: target type
            - task
            - algo
            - dataset
            - model
    """
    info_str = osp.splitext(
        osp.relpath(path, osp.join(MMPOSE_ROOT, 'configs')))[0]

    target, task, algorithm, dataset, model = (info_str.split(osp.sep) +
                                               ['None'] * 5)[:5]

    # capitalize target
    target = target.capitalize()
    # convert task name to readable version
    task2readable = {
        '2d_kpt_sview_rgb_img': '2D Keypoint',
        '2d_kpt_sview_rgb_vid': '2D Keypoint',
        '3d_kpt_sview_rgb_img': '3D Keypoint',
        '3d_kpt_mview_rgb_img': '3D Keypoint',
        '3d_kpt_sview_rgb_vid': '3D Keypoint',
        '3d_mesh_sview_rgb_img': '3D Mesh',
        'gesture_sview_rgbd_vid': 'Gesture',
        None: None
    }
    task_readable = task2readable.get(task)

    model_info = {
        'target': target,
        'task': task_readable,
        'algorithm': algorithm,
        'dataset': dataset,
        'model': model,
    }

    return model_info


def parse_md(md_file):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file: Path to .md file.
    Returns:
        Bool: If the target YAML file is different from the original.
    """

    collection = {'Name': None, 'Paper': None}
    models = []

    # get readme files
    collection2readme = collect_paper_readme()

    # record the publish year of the latest paper
    paper_year = -1
    dataset = None

    # architectures for collection and model
    architecture = []

    with open(md_file, 'r', encoding='utf-8') as md:
        lines = md.readlines()
        i = 0
        while i < len(lines):

            # parse reference
            if lines[i][:2] == '<!':
                details_start = lines.index('<details>\n', i)
                details_end = lines.index('</details>\n', i)
                details = ''.join(lines[details_start:details_end + 1])
                url, name, year = re.findall(
                    r'<a href="(.*)">(.*?)[ ]*\(.*\'(.*)\).*</a>', details)[0]
                year = int(year)
                try:
                    title = re.findall(r'title.*\{(.*)\}\,', details)[0]
                except IndexError:
                    title = None

                paper_type = re.findall(r'\[(.*)\]', lines[i])[0]

                # lower priority for dataset paper
                if paper_type == 'DATASET':
                    year = 0

                if year > paper_year:
                    collection['Paper'] = dict(Title=title, URL=url)
                    collection['Name'] = name
                    collection['README'] = collection2readme[name]
                    paper_year = year

                # get architecture
                if paper_type in {'ALGORITHM', 'BACKBONE'}:
                    architecture.append(name)

                # get dataset
                elif paper_type == 'DATASET':
                    dataset = name

                i = details_end + 1

            # parse table
            elif lines[i][0] == '|' and i + 1 < len(lines) and \
                    lines[i + 1][:3] == '| :':
                cols = [col.strip() for col in lines[i].split('|')][1:-1]
                config_idx = cols.index('Arch')
                ckpt_idx = cols.index('ckpt')
                try:
                    flops_idx = cols.index('FLOPs')
                except ValueError:
                    flops_idx = -1
                try:
                    params_idx = cols.index('Params')
                except ValueError:
                    params_idx = -1
                metric_name_list, metric_idx_list = collate_metrics(cols)

                j = i + 2
                while j < len(lines) and lines[j][0] == '|':
                    line = lines[j].split('|')[1:-1]

                    if line[config_idx].find('](') == -1:
                        j += 1
                        continue
                    left = line[config_idx].index('](') + 2
                    right = line[config_idx].index(')', left)
                    config = line[config_idx][left:right].strip('./')

                    left = line[ckpt_idx].index('](') + 2
                    right = line[ckpt_idx].index(')', left)
                    ckpt = line[ckpt_idx][left:right]

                    model_info = parse_config_path(config)
                    model_name = '_'.join(
                        [model_info['algorithm'], model_info['model']])
                    task_name = ' '.join(
                        [model_info['target'], model_info['task']])

                    metadata = {
                        'Training Data': dataset,
                        'Architecture': architecture
                    }
                    if flops_idx != -1:
                        metadata['FLOPs'] = float(line[flops_idx])
                    if params_idx != -1:
                        metadata['Parameters'] = float(line[params_idx])

                    metrics = {}
                    for metric_name, metric_idx in zip(metric_name_list,
                                                       metric_idx_list):
                        metrics[metric_name] = float(line[metric_idx])

                    model = {
                        'Name':
                        model_name,
                        'In Collection':
                        collection['Name'],
                        'Config':
                        config,
                        'Metadata':
                        metadata,
                        'Results': [{
                            'Task': task_name,
                            'Dataset': dataset,
                            'Metrics': metrics
                        }],
                        'Weights':
                        ckpt
                    }
                    models.append(model)
                    j += 1
                i = j

            else:
                i += 1

    result = {'Collections': [collection], 'Models': models}
    yml_file = osp.splitext(md_file)[0] + '.yml'
    is_different = dump_yaml_and_check_difference(result, yml_file)
    return is_different


def update_model_index():
    """Update model-index.yml according to model .md files.

    Returns:
        Bool: If the updated model-index.yml is different from the original.
    """
    configs_dir = osp.join(MMPOSE_ROOT, 'configs')
    yml_files = glob.glob(osp.join(configs_dir, '**', '*.yml'), recursive=True)
    yml_files.sort()

    model_index = {
        'Import': [
            osp.relpath(yml_file, MMPOSE_ROOT).replace('\\', '/')
            for yml_file in yml_files
        ]
    }
    model_index_file = osp.join(MMPOSE_ROOT, 'model-index.yml')
    is_different = dump_yaml_and_check_difference(model_index,
                                                  model_index_file)

    return is_different


if __name__ == '__main__':

    file_list = [
        fn for fn in sys.argv[1:]
        if osp.basename(fn) != 'README.md' and '_base_' not in fn
    ]

    if not file_list:
        sys.exit(0)

    file_modified = False
    for fn in file_list:
        file_modified |= parse_md(fn)

    file_modified |= update_model_index()

    sys.exit(1 if file_modified else 0)
