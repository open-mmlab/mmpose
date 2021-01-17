import numpy as np
import torch

from .. import builder
from ..mesh_heads.discriminator import SMPLDiscriminator
from ..registry import POSENETS
from .base import BasePose

try:
    from smplx import SMPL
    has_smpl = True
except (ImportError, ModuleNotFoundError):
    has_smpl = False


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


@POSENETS.register_module()
class ParametricMesh(BasePose):
    """Model-based 3D human mesh detector. Take a single color image as input
    and output 3D joints, SMPL parameters and camera parameters.

    Args:
        backbone (dict): Backbone modules to extract feature.
        mesh_head (dict): Mesh head to process feature.
        smpl (dict): Config for SMPL model.
        disc (dict): Discriminator for SMPL parameters. Default: None.
        loss_gan (dict): Config for adversarial loss. Default: None.
        loss_mesh (dict): Config for mesh loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
    """

    def __init__(self,
                 backbone,
                 mesh_head,
                 smpl,
                 disc=None,
                 loss_gan=None,
                 loss_mesh=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        assert has_smpl, 'Please install smplx to use SMPL.'

        self.backbone = builder.build_backbone(backbone)
        self.mesh_head = builder.build_head(mesh_head)
        self.generator = torch.nn.Sequential(self.backbone, self.mesh_head)

        self.smpl = SMPL(
            model_path=smpl['smpl_path'],
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False)

        joints_regressor = torch.tensor(
            np.load(smpl['joints_regressor']), dtype=torch.float).unsqueeze(0)
        self.register_buffer('joints_regressor', joints_regressor)

        self.with_gan = disc is not None and loss_gan is not None
        if self.with_gan:
            self.discriminator = SMPLDiscriminator(**disc)
            self.loss_gan = builder.build_loss(loss_gan)
        self.disc_step_count = 0

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_mesh = builder.build_loss(loss_mesh)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        self.mesh_head.init_weights()
        if self.with_gan:
            self.discriminator.init_weights()

    def train_step(self, data_batch, optimizer, **kwargs):
        """Train step function.

        In this function, the detector will finish the train step following
        the pipeline:
        1. get fake and real SMPL parameters
        2. optimize discriminator (if have)
        3. optimize generator

        If `self.train_cfg.disc_step > 1`, the train step will contain multiple
        iterations for optimizing discriminator with different input data and
        only one iteration for optimizing generator after `disc_step`
        iterations for discriminator.

        Args:
            data_batch (torch.Tensor): Batch of data as input.
            optimizer (dict[torch.optim.Optimizer]): Dict with optimizers for
                generator and discriminator (if have).

        Returns:
            outputs (dict): Dict with loss, information for logger,
            the number of samples.
        """

        img = data_batch['img']
        pred_smpl = self.generator(img)
        pred_pose, pred_beta, pred_camera = pred_smpl

        # optimize discriminator (if have)
        if self.train_cfg['disc_step'] > 0 and self.with_gan:
            set_requires_grad(self.discriminator, True)
            fake_data = (pred_camera.detach(), pred_pose.detach(),
                         pred_beta.detach())
            mosh_theta = data_batch['mosh_theta']
            real_data = (mosh_theta[:, :3], mosh_theta[:,
                                                       3:75], mosh_theta[:,
                                                                         75:])
            fake_score = self.discriminator(fake_data)
            real_score = self.discriminator(real_data)

            disc_losses = {}
            disc_losses['real_loss'] = self.loss_gan(
                real_score, target_is_real=True, is_disc=True)
            disc_losses['fake_loss'] = self.loss_gan(
                fake_score, target_is_real=False, is_disc=True)
            loss_disc, log_vars_d = self._parse_losses(disc_losses)

            optimizer['discriminator'].zero_grad()
            loss_disc.backward()
            optimizer['discriminator'].step()
            self.disc_step_count = \
                (self.disc_step_count + 1) % self.train_cfg['disc_step']

            if self.disc_step_count != 0:
                outputs = dict(
                    loss=loss_disc,
                    log_vars=log_vars_d,
                    num_samples=len(next(iter(data_batch.values()))))
                return outputs

        # optimize generator
        pred_out = self.smpl(
            betas=pred_beta,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, :1],
            pose2rot=False)
        pred_vertices = pred_out.vertices
        pred_joints_3d = self.get_3d_joints_from_mesh(pred_vertices)
        gt_beta = data_batch['beta']
        gt_pose = data_batch['pose']
        gt_vertices = self.smpl(
            betas=gt_beta,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]).vertices
        pred = dict(
            pose=pred_pose,
            beta=pred_beta,
            camera=pred_camera,
            vertices=pred_vertices,
            joints_3d=pred_joints_3d)

        target = {
            key: data_batch[key]
            for key in [
                'pose', 'beta', 'has_smpl', 'joints_3d', 'joints_2d',
                'joints_3d_visible', 'joints_2d_visible'
            ]
        }
        target['vertices'] = gt_vertices

        losses = self.loss_mesh(pred, target)

        if self.with_gan:
            set_requires_grad(self.discriminator, False)
            pred_theta = (pred_camera, pred_pose, pred_beta)
            pred_score = self.discriminator(pred_theta)
            loss_adv = self.loss_gan(
                pred_score, target_is_real=True, is_disc=False)
            losses['adv_loss'] = loss_adv

        loss, log_vars = self._parse_losses(losses)
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def forward_train(self, *args, **kwargs):
        """Forward function for training.

        For ParametricMesh, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def val_step(self, data_batch, **kwargs):
        """Forward function for evaluation.

        Args:
            data_batch (dict): Contain data for forward.

        Returns:
            dict: Contain the results from model.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.generator(img)
        return output

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == 1
        assert len(img_metas) == 1

        pred_smpl = self.generator(img)
        pred_pose, pred_beta, pred_camera = pred_smpl
        pred_out = self.smpl(
            betas=pred_beta,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, :1],
            pose2rot=False)
        pred_vertices = pred_out.vertices
        pred_joints_3d = self.get_3d_joints_from_mesh(pred_vertices)

        all_preds = (pred_joints_3d.detach().cpu().numpy(),
                     (pred_pose.detach().cpu().numpy(),
                      pred_beta.detach().cpu().numpy()),
                     pred_camera.detach().cpu().numpy())

        all_boxes = np.zeros((1, 6), dtype=np.float32)
        image_path = []

        img_metas = img_metas[0]
        c = img_metas['center'].reshape(1, -1)
        s = img_metas['scale'].reshape(1, -1)

        score = 1.0
        if 'bbox_score' in img_metas:
            score = np.array(img_metas['bbox_score']).reshape(-1)

        all_boxes[0, 0:2] = c[:, 0:2]
        all_boxes[0, 2:4] = s[:, 0:2]
        all_boxes[0, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[0, 5] = score
        image_path.extend(img_metas['image_file'])

        return all_preds, all_boxes, image_path

    def get_3d_joints_from_mesh(self, vertices):
        """Get 3D joints from 3D mesh using predefined joints regressor."""
        return torch.matmul(
            self.joints_regressor.to(vertices.device), vertices)

    def forward(self, img, img_metas=None, return_loss=False, **kwargs):
        """Forward function.

        Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note:
            batch_size: N
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW

        Args:
            img (torch.Tensor[N x C x imgH x imgW]): Input images.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            Return predicted 3D joints, SMPL parameters, boxes and image paths.
        """

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        return self.forward_test(img, img_metas, **kwargs)
