# import necessary libraries
import torch
import torch.nn as nn


class WeightedL1Loss(nn.L1Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input_0, input_1 = torch.masked_select(input, target == 0), torch.masked_select(input, target != 0)
        target_0, target_1 = torch.masked_select(target, target == 0), torch.masked_select(target, target != 0)

        loss_0, loss_1 = super().forward(input_0, target_0), super().forward(input_1, target_1)
        if torch.isnan(loss_0).any():
            return loss_1 / 2
        elif torch.isnan(loss_1).any():
            return loss_0 / 2
        else:
            return (loss_0 + loss_1) / 2


class NestedBoundaryLoss(WeightedL1Loss):
    def __init__(self):
        super().__init__()
        self.classifier_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        loss_inner = super(WeightedL1Loss, self).forward(input[..., 0], target[..., 0])
        loss_outer = super(NestedBoundaryLoss, self).forward(input[..., 1], target[..., 1])
        loss_total = super(WeightedL1Loss, self).forward(input[..., :2].sum(dim=1), target[..., :2].sum(dim=1))

        myo_target = target[..., 1] != 0
        loss_cls = self.classifier_loss(input[..., 2], myo_target.float())
        return loss_inner, loss_outer, loss_total, loss_cls
