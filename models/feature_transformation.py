import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, x):

        i2_part = x.view(-1, 4, 100, 256)          # [b, 4, 100, 256]
        i2_part_fi = torch.max(i2_part, 2)[0]      # [b, 4, 256]   Global features of a single image
        i2_part_fgi1 = torch.sum(i2_part_fi, 1)     # [b, 256]     Combined features of the four images
        i2_part_fgi2 = i2_part_fgi1.unsqueeze(1).repeat(1, 4, 1)   # [b, 4, 256]

        x = self.encoder(i2_part_fi)
        x = self.decoder(x)
        return x, i2_part_fgi2





