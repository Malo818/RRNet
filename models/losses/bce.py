import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, batch_size):
        super(BCELoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, global_score, n):

        bce_loss = nn.BCELoss()

        if n<= (256//global_score.size(0)):
            label = torch.ones(global_score.shape[0]).cuda()
        else:
            pos_label = torch.ones(global_score.shape[0]//2).cuda()
            neg_label = torch.zeros(global_score.shape[0]//2).cuda()
            label = torch.cat((pos_label, neg_label), 0)

        loss = bce_loss(global_score.squeeze(), label)
        return loss