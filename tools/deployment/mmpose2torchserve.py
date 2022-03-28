# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from argparse import ArgumentParser, Namespace
from tempfile import TemporaryDirectory

import mmcv
import torch
from mmcv.runner import CheckpointLoader

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    package_model = None


def mmpose2torchserve(config_file: str,
                      checkpoint_file: str,
                      output_folder: str,
                      model_name: str,
                      model_version: str = '1.0',
                      force: bool = False):
    """Converts MMPose model (config + checkpoint) to TorchServe `.mar`.

    Args:
        config_file:
            In MMPose config format.
            The contents vary for each task repository.
        checkpoint_file:
            In MMPose checkpoint format.
            The contents vary for each task repository.
        output_folder:
            Folder where `{model_name}.mar` will be created.
            The file created will be in TorchServe archive format.
        model_name:
            If not None, used for naming the `{model_name}.mar` file
            that will be created under `output_folder`.
            If None, `{Path(checkpoint_file).stem}` will be used.
        model_version:
            Model's version.
        force:
            If True, if there is an existing `{model_name}.mar`
            file under `output_folder` it will be overwritten.
    """

    mmcv.mkdir_or_exist(output_folder)

    config = mmcv.Config.fromfile(config_file)

    with TemporaryDirectory() as tmpdir:
        model_file = osp.join(tmpdir, 'config.py')
        config.dump(model_file)
        handler_path = osp.join(osp.dirname(__file__), 'mmpose_handler.py')
        model_name = model_name or osp.splitext(
            osp.basename(checkpoint_file))[0]

        # use mmcv CheckpointLoader if checkpoint is not from a local file
        if not osp.isfile(checkpoint_file):
            ckpt = CheckpointLoader.load_checkpoint(checkpoint_file)
            checkpoint_file = osp.join(tmpdir, 'checkpoint.pth')
            with open(checkpoint_file, 'wb') as f:
                torch.save(ckpt, f)

        args = Namespace(
            **{
                'model_file': model_file,
                'serialized_file': checkpoint_file,
                'handler': handler_path,
                'model_name': model_name,
                'version': model_version,
                'export_path': output_folder,
                'force': force,
                'requirements_file': None,
                'extra_files': None,
                'runtime': 'python',
                'archive_format': 'default'
            })
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


def parse_args():
    parser = ArgumentParser(
        description='Convert MMPose models to TorchServe `.mar` format.')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Folder where `{model_name}.mar` will be created.')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='If not None, used for naming the `{model_name}.mar`'
        'file that will be created under `output_folder`.'
        'If None, `{Path(checkpoint_file).stem}` will be used.')
    parser.add_argument(
        '--model-version',
        type=str,
        default='1.0',
        help='Number used for versioning.')
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='overwrite the existing `{model_name}.mar`')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    if package_model is None:
        raise ImportError('`torch-model-archiver` is required.'
                          'Try: pip install torch-model-archiver')

    mmpose2torchserve(args.config, args.checkpoint, args.output_folder,
                      args.model_name, args.model_version, args.force)
