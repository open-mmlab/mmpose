import argparse

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from tools.export_specs import export_for_lv
from tools.pytorch2onnx import pytorch2onnx, _convert_batchnorm

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--add-normalization',
        action='store_true',
        help='add normalization layer to the exported onnx model')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 256, 192],
        help='input size')
    parser.add_argument(
        '--project_name',
        type=str,
        help='name of the project for lv usage, should match the folder name structure'
             'in the ml_models repo: e.g. \"brummer\"',
        required=True
    )
    parser.add_argument(
        '--author',
        type=str,
        help='full name of the Author of this training: e.g. \"Christian Holland\"',
        required=True
    )
    parser.add_argument(
        '--jira_task',
        type=str,
        help='shortened name of the Jira task for this training: e.g. \"OR-1926\"',
        required=True
    )
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMPose only supports opset 11 now'

    cfg = mmcv.Config.fromfile(args.config)
    # build the model
    model = build_posenet(cfg.model)
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model_output_path = export_for_lv(args)

    pytorch2onnx(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=model_output_path,
        verify=args.verify,
        add_normalization=args.add_normalization)
    print(f"Model exported successfully to: {model_output_path}")