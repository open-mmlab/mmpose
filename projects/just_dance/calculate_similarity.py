import numpy as np
import torch

flip_indices = np.array(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
valid_indices = np.array([0] + list(range(5, 17)))


@torch.no_grad()
def _calculate_similarity(tch_kpts: np.ndarray, stu_kpts: np.ndarray):

    stu_kpts = torch.from_numpy(stu_kpts[:, None, valid_indices])
    tch_kpts = torch.from_numpy(tch_kpts[None, :, valid_indices])
    stu_kpts = stu_kpts.expand(stu_kpts.shape[0], tch_kpts.shape[1],
                               stu_kpts.shape[2], 3)
    tch_kpts = tch_kpts.expand(stu_kpts.shape[0], tch_kpts.shape[1],
                               stu_kpts.shape[2], 3)

    matrix = torch.stack((stu_kpts, tch_kpts), dim=4)
    if torch.cuda.is_available():
        matrix = matrix.cuda()
    mask = torch.logical_and(matrix[:, :, :, 2, 0] > 0.3,
                             matrix[:, :, :, 2, 1] > 0.3)
    matrix[~mask] = 0.0

    matrix_ = matrix.clone()
    matrix_[matrix == 0] = 256
    x_min = matrix_.narrow(3, 0, 1).min(dim=2).values
    y_min = matrix_.narrow(3, 1, 1).min(dim=2).values
    matrix_ = matrix.clone()
    # matrix_[matrix == 0] = 0
    x_max = matrix_.narrow(3, 0, 1).max(dim=2).values
    y_max = matrix_.narrow(3, 1, 1).max(dim=2).values

    matrix_ = matrix.clone()
    matrix_[:, :, :, 0] = (matrix_[:, :, :, 0] - x_min) / (
        x_max - x_min + 1e-4)
    matrix_[:, :, :, 1] = (matrix_[:, :, :, 1] - y_min) / (
        y_max - y_min + 1e-4)
    matrix_[:, :, :, 2] = (matrix_[:, :, :, 2] > 0.3).float()
    xy_dist = matrix_[..., :2, 0] - matrix_[..., :2, 1]
    score = matrix_[..., 2, 0] * matrix_[..., 2, 1]

    similarity = (torch.exp(-50 * xy_dist.pow(2).sum(dim=-1)) *
                  score).sum(dim=-1) / (
                      score.sum(dim=-1) + 1e-6)
    num_visible_kpts = score.sum(dim=-1)
    similarity = similarity * torch.log(
        (1 + (num_visible_kpts - 1) * 10).clamp(min=1)) / np.log(161)

    similarity[similarity.isnan()] = 0

    return similarity


@torch.no_grad()
def calculate_similarity(tch_kpts: np.ndarray, stu_kpts: np.ndarray):
    assert tch_kpts.shape[1] == 17
    assert tch_kpts.shape[2] == 3
    assert stu_kpts.shape[1] == 17
    assert stu_kpts.shape[2] == 3

    similarity1 = _calculate_similarity(tch_kpts, stu_kpts)

    stu_kpts_flip = stu_kpts[:, flip_indices]
    stu_kpts_flip[..., 0] = 191.5 - stu_kpts_flip[..., 0]
    similarity2 = _calculate_similarity(tch_kpts, stu_kpts_flip)

    similarity = torch.stack((similarity1, similarity2)).max(dim=0).values

    return similarity


@torch.no_grad()
def select_piece_from_similarity(similarity):
    m, n = similarity.size()
    row_indices = torch.arange(m).view(-1, 1).expand(m, n).to(similarity)
    col_indices = torch.arange(n).view(1, -1).expand(m, n).to(similarity)
    diagonal_indices = similarity.size(0) - 1 - row_indices + col_indices
    unique_diagonal_indices, inverse_indices = torch.unique(
        diagonal_indices, return_inverse=True)

    diagonal_sums_list = torch.zeros(
        unique_diagonal_indices.size(0),
        dtype=similarity.dtype,
        device=similarity.device)
    diagonal_sums_list.scatter_add_(0, inverse_indices.view(-1),
                                    similarity.view(-1))
    diagonal_sums_list[:min(m, n) // 4] = 0
    diagonal_sums_list[-min(m, n) // 4:] = 0
    index = diagonal_sums_list.argmax().item()

    similarity_smooth = torch.nn.functional.max_pool2d(
        similarity[None], (1, 11), stride=(1, 1), padding=(0, 5))[0]
    similarity_vec = similarity_smooth.diagonal(offset=index - m +
                                                1).cpu().numpy()

    stu_start = max(0, m - 1 - index)
    tch_start = max(0, index - m + 1)

    return dict(
        stu_start=stu_start,
        tch_start=tch_start,
        length=len(similarity_vec),
        similarity=similarity_vec)
