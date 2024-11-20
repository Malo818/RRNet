import torch
import torch.nn as nn
from einops import rearrange

from .imagenet import *
from .pointnet import *
from .transformer import *
from .feature_transformation import Autoencoder

class RRNet(nn.Module):
    def __init__(self,embedding_size):
        super(RRNet, self).__init__()
        self.embedding_size=embedding_size

        self.img_feature_extraction = ResNet34()
        self.pc_feature_extraction = PcEncoding()

        self.img_pos_encoding = PositionEmbedding(2, 256)
        self.feature_fusion = FeatureFusion(D_MODEL=256, NHEAD=8, LAYER_NAMES=['Sa', 'SCa', 'MCa'] * 4, ATTENTION = 'full' )

        self.ae = Autoencoder()
        self.pc_feature_transformation = PcDecoding()

    def forward(self,img, pc):

        # Feature Extraction
        bs = img.size(0)//4
        i1 = self.img_feature_extraction(img)                                           # i1[4bs,256,10,10]
        fe_xyz, p1, Coded_set = self.pc_feature_extraction(pc)                          # p1[mbs,100,256]

        # Feature Fusion
        img_x, img_y = torch.meshgrid(torch.arange(0, 10), torch.arange(0, 10), indexing='ij')
        img_xy = rearrange(torch.cat((img_x.unsqueeze(-1), img_y.unsqueeze(-1)), dim=2).expand(1, 10, 10, 2), 'b h w d -> b (h w) d').cuda()
        img_pos = self.img_pos_encoding(img_xy)
        i1_pos = rearrange(i1, 'b c h w -> b (h w) c') + img_pos              # Image position coding   [4bs,100,256]
        i2, p2 = self.feature_fusion(i1_pos, p1)

        # Feature transformation
        fgi, fgi4 = self.ae(i2)                                 # fgi[b, 4, 256]
        fgp0 = torch.max(p2.view(bs, -1, 100, 256), 2)[0]       # [b, m, 256]
        fgp = torch.sum(fgp0, 1)                                # [b, 256]

        Coded_set.append(([fe_xyz, p2]))

        fp = self.pc_feature_transformation(Coded_set)

        return fgi, fgi4, fgp, fp


