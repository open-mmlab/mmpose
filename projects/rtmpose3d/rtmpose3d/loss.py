from mmpose.models.losses import KLDiscretLoss
from mmpose.registry import MODELS


@MODELS.register_module()
class KLDiscretLossWithWeight(KLDiscretLoss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss_name = 'loss_kld'

    def forward(self, pred_simcc, gt_simcc, target_weight):
        N, K, _ = pred_simcc[0].shape
        loss = 0

        for pred, target, weight in zip(pred_simcc, gt_simcc, target_weight):
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1, target.size(-1))
            weight = weight.reshape(-1)

            t_loss = self.criterion(pred, target).mul(weight)

            if self.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, self.mask] = t_loss[:, self.mask] * self.mask_weight

            loss = loss + t_loss.sum()

        return loss / K

    @property
    def loss_name(self):
        """Loss Name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
