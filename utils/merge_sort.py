import torch
import random
import numpy as np
import torch.nn as nn

class lobalclassification(nn.Module):
    def __init__(self):
        super(lobalclassification, self).__init__()
        self.lclsort = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y1, y2):
        y1_repeat = y1.unsqueeze(1).repeat(1, 4, 1)
        x = x.view(-1, 256)
        y1_repeat = y1_repeat.view(-1, 256)

        if len(y2)<=(32//y1.size(0)):
            x_repeat = x.repeat(2, 1)
            y1_rerepeat = y1_repeat.repeat(2, 1)
            p_global_connection = torch.cat([x_repeat, y1_rerepeat], dim=1)
            out = self.lclsort(p_global_connection)
        else:
           random_element = random.choice(y2[:(32//y1.size(0))*(len(y2)//(32//y1.size(0)))])
           random_element_repeat = random_element.unsqueeze(1).repeat(1, 4, 1).detach()
           random_element_repeat = random_element_repeat.view(-1, 256)
           x_repeat = x.repeat(2, 1)
           y1y2 = torch.cat([y1_repeat, random_element_repeat], dim=0)
           pn_global_connection = torch.cat([x_repeat, y1y2], dim=1)
           out = self.lclsort(pn_global_connection)
        return out


class pointsclassification(nn.Module):
    def __init__(self):
        super(pointsclassification, self).__init__()
        self.plcsort = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        bs = x.size(0)
        x_repeat = x.unsqueeze(2).repeat(1, 1, y.size(1), 1)
        x_rerepeat = x_repeat.unsqueeze(1).repeat(1, y.size(0)//bs, 1, 1, 1)

        y_repeat = y.view(bs, -1, y.size(1), y.size(2)).unsqueeze(2).repeat(1, 1, 4, 1, 1)
        concatenated_features = torch.cat((y_repeat, x_rerepeat), -1)

        x = self.plcsort(concatenated_features)

        return x.view(-1, 4, x.size(3))


