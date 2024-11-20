import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_distance_squared(x, y):
    xx = torch.sum(torch.pow(x, 2), 1).view(-1, 1)
    yy = torch.sum(torch.pow(y, 2), 1).view(1, -1)
    pdist = xx + yy - 2.0 * torch.mm(x, torch.t(y))
    return pdist

class HardTripletLoss(nn.Module):
    def __init__(self, margin=0.2, hardest=False):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest

    def forward(self, x, y):
        x1 = x.view(-1, 256)
        y1 = y.view(-1, 256)
        batch_size = x.shape[0]
        pdist = _pairwise_distance_squared(x1, y1)

        if self.hardest:
            diag = torch.arange(0, pdist.size(0))
            diag = diag.to(x.device)
            d_pos = pdist[diag, diag]

            mask = torch.zeros(pdist.size(0), pdist.size(1), dtype=torch.bool)
            for i in range(batch_size):
                start_row = i * 4
                end_row = (i + 1) * 4
                start_col = i * 4
                end_col = (i + 1) * 4
                mask[start_row:end_row, start_col:end_col] = True
            pdist[mask] = float("Inf")
            d_neg, _ = torch.min(pdist, 1)
            loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
            loss = torch.mean(loss)
        else:
            diag = torch.arange(0, batch_size)
            diag = diag.to(x.device)
            d_pos = pdist[diag, diag].unsqueeze(1)
            loss = d_pos - pdist + self.margin
            loss = torch.clamp(loss, min=0.0)
            loss[diag, diag] = 0.0
            loss = torch.mean(loss)
        return loss
