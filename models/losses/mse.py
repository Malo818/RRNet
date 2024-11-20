import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        # TODO: repsect reduction rule
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        y_repeat = y.unsqueeze(1).repeat(1, 4, 1)
        loss = torch.pow(x - y_repeat, 2)
        return loss.mean()