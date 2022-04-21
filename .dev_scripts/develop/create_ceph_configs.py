import glob
import os.path as osp
import re
import shutil
from argparse import ArgumentParser
from functools import wraps
from multiprocessing.dummy import Pool


def bundle_args(func):

    @wraps(func)
    def wrapped(args):
        return func(*args)

    return wrapped


@bundle_args
def update_ceph_config(filename, args, dry_run=False):
    if filename.startswith(osp.join('configs_ceph', '_base_')):
        # Skip base configs
        return None
    # try:
    with open(filename) as f:
        content = f.read()

    work_dir = f'\'sh1984:s3://{args.bucket}/{args.work_dir_prefix}/\''
    ceph_config = ('# ceph configs\n'
                   'file_client_args = dict('
                   'backend=\'petrel\','
                   'path_mapping={'
                   '\'.data/\': \'openmmlab:s3://openmmlab/datasets/pose/\', '
                   '\'data/\': \'openmmlab:s3://openmmlab/datasets/pose/\''
                   '})\n')
    try:
        # Update evaluation configs
        match = re.search(r'evaluation = dict\(', content)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + f'out_dir={work_dir}, ' + content[
                insert_pos:]
        else:
            ceph_config += f'evaluation = dict(out_dir={work_dir})\n'

        # Update checkpoint configs
        match = re.search(r'checkpoint_config = dict\(', content, re.S)
        if match:
            insert_pos = match.end()
            content = (
                content[:insert_pos] +
                f'max_keep_ckpts=2, out_dir={work_dir}, ' +
                content[insert_pos:])
        else:
            ceph_config += ('checkpoint_config = dict(max_keep_ckpts=2, '
                            f'out_dir={work_dir})\n')

        # Update log configs
        match = re.search(r'dict\(.*?type=\'TextLoggerHook\'', content, re.S)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + f'out_dir={work_dir}' + content[
                insert_pos:]
        else:
            content += ('log_config = dict(hooks=[dict(type=\'TextLoggerHook'
                        f'\', out_dir={work_dir})])')

        # Update image loading in pipelines
        content = re.sub(
            r'dict\(type=\'LoadImageFromFile\'\)',
            'dict(type=\'LoadImageFromFile\', '
            'file_client_args=file_client_args)',
            content,
            flags=re.S)

        # Update pre-trained model path
        content = re.sub(
            r'(?<=pretrained=\')https://download\.openmmlab\.com',
            'openmmlab:s3://openmmlab/checkpoints',
            content,
            flags=re.S)

        # Add ceph config
        insert_pos = 0
        match = re.search(r'_base_ = \[.*?\]\n', content, re.S)
        if match:
            # Insert Ceph configs after _base_
            insert_pos = match.end()
            content = content[:insert_pos] + ceph_config + content[insert_pos:]

        if not dry_run:
            with open(filename, 'w') as f:
                f.write(content)

        return True

    except:  # noqa
        if dry_run:
            raise
        else:
            return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('bucket', type=str, help='')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Thread number to process files')
    parser.add_argument(
        '--work-dir-prefix',
        type=str,
        default='work_dirs',
        help='Default prefix of the work dirs in the bucket')
    parser.add_argument(
        '--test-file', type=str, default=None, help='Dry-run on a test file.')

    args = parser.parse_args()

    if args.test_file is None:

        print('Copying config files to "config_ceph" ...')
        shutil.copytree('configs', 'configs_ceph', dirs_exist_ok=True)

        print('Updating ceph configuration ...')
        with Pool(processes=8) as pool:
            files = glob.glob(
                osp.join('configs_ceph', '**', '*.py'), recursive=True)
            res = pool.map(update_ceph_config, [(fn, args) for fn in files])
            res = list(res)

        count_skip = res.count(None)
        count_done = res.count(True)
        count_fail = res.count(False)
        fail_list = [fn for status, fn in zip(res, files) if status is False]

        print(f'Successfully update {count_done} configs.')
        print(f'Skip {count_skip} configs.')
        print(f'Fail {count_fail} configs:')
        for fn in fail_list:
            print(fn)

    else:
        update_ceph_config((args.test_file, args, True))
