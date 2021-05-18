import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from torch.nn import functional as F

from ..registry import HEADS


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion
            -- size = [B, 3, 3]
    """
    l2norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l2norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion
            -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1],\
        norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(B, 3, 3)
    return rotMat


@HEADS.register_module()
class HMRMeshHead(nn.Module):
    """SMPL parameters regressor head of simple baseline paper
    ref: Angjoo Kanazawa. ``End-to-end Recovery of Human Shape and Pose''.

    Args:
        in_channels (int): Number of input channels
        in_res (int): The resolution of input feature map.
        smpl_mean_parameters (str): The file name of the mean SMPL parameters
        n_iter (int): The iterations of estimating delta parameters
    """

    def __init__(self, in_channels, smpl_mean_params=None, n_iter=3):
        super().__init__()

        self.in_channels = in_channels
        self.n_iter = n_iter

        npose = 24 * 6
        nbeta = 10
        ncam = 3
        hidden_dim = 1024

        self.fc1 = nn.Linear(in_channels + npose + nbeta + ncam, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hidden_dim, npose)
        self.decshape = nn.Linear(hidden_dim, nbeta)
        self.deccam = nn.Linear(hidden_dim, ncam)

        # Load mean SMPL parameters
        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(
                mean_params['pose'][:]).unsqueeze(0).float()
            init_shape = torch.from_numpy(
                mean_params['shape'][:]).unsqueeze(0).float()
            init_cam = torch.from_numpy(
                mean_params['cam']).unsqueeze(0).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x):
        """Forward function.

        x is the image feature map and is expected to be in shape (batch size x
        channel number x height x width)
        """
        batch_size = x.shape[0]
        # extract the global feature vector by average along
        # spatial dimension.
        x = x.mean(dim=-1).mean(dim=-1)

        init_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for _ in range(self.n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        out = (pred_rotmat, pred_shape, pred_cam)
        return out

    def init_weights(self):
        """Initialize model weights."""
        xavier_init(self.decpose, gain=0.01)
        xavier_init(self.decshape, gain=0.01)
        xavier_init(self.deccam, gain=0.01)
