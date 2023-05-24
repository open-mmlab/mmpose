import argparse

import torch
import torch.nn.functional as F
from mmdet.apis import init_detector
from torch import nn


def build_model_from_cfg(config_path: str, checkpoint_path: str, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


class RTMDet(nn.Module):
    """Load RTMDet model and add postprocess.

    Args:
        model (nn.Module): The RTMDet model.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.stage = [80, 40, 20]
        self.input_shape = 640

    def forward(self, inputs):
        """model forward function."""
        boxes = []
        neck_outputs = self.model(inputs)
        for i, (cls, box) in enumerate(zip(*neck_outputs)):
            cls = cls.permute(0, 2, 3, 1)
            box = box.permute(0, 2, 3, 1)
            box = self.decode(box, cls, i)
            boxes.append(box)
        result_box = torch.cat(boxes, dim=1)
        return result_box

    def decode(self, box: torch.Tensor, cls: torch.Tensor, stage: int):
        """RTMDet postprocess function.

        Args:
            box (torch.Tensor): output boxes.
            cls (torch.Tensor): output cls.
            stage (int): RTMDet output stage.

        Returns:
            torch.Tensor: The decode boxes.
                Format is [x1, y1, x2, y2, class, confidence]
        """
        cls = F.sigmoid(cls)
        conf = torch.max(cls, dim=3, keepdim=True)[0]
        cls = torch.argmax(cls, dim=3, keepdim=True).to(torch.float32)

        box = torch.cat([box, cls, conf], dim=-1)

        step = self.input_shape // self.stage[stage]

        block_step = torch.linspace(
            0, self.stage[stage] - 1, steps=self.stage[stage],
            device='cuda') * step
        block_x = torch.broadcast_to(block_step,
                                     [self.stage[stage], self.stage[stage]])
        block_y = torch.transpose(block_x, 1, 0)
        block_x = torch.unsqueeze(block_x, 0)
        block_y = torch.unsqueeze(block_y, 0)
        block = torch.stack([block_x, block_y], -1)

        box[..., :2] = block - box[..., :2]
        box[..., 2:4] = block + box[..., 2:4]
        box = box.reshape(1, -1, 6)
        return box


def parse_args():
    parser = argparse.ArgumentParser(
        description='convert rtmdet model to ONNX.')
    parser.add_argument(
        '--config', type=str, help='rtmdet config file path from mmdetection.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='rtmdet checkpoint path from mmdetection.')
    parser.add_argument('--output', type=str, help='output filename.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '--input-name', type=str, default='image', help='ONNX input name.')
    parser.add_argument(
        '--output-name', type=str, default='output', help='ONNX output name.')
    parser.add_argument(
        '--opset', type=int, default=11, help='ONNX opset version.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model = build_model_from_cfg(args.config, args.checkpoint, args.device)
    rtmdet = RTMDet(model)
    rtmdet.eval()
    x = torch.randn((1, 3, 640, 640), device=args.device)

    torch.onnx.export(
        rtmdet,
        x,
        args.output,
        input_names=[args.input_name],
        output_names=[args.output_name],
        opset_version=args.opset)
