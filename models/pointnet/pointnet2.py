import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PcEncoding(nn.Module):
    def __init__(self):
        super(PcEncoding, self).__init__()
        self.sa1 = PointNetSetAbstraction(3350, 0.1, 32, 3+3, [32, 32, 32])
        self.sa2 = PointNetSetAbstraction(1100, 0.3, 32, 32+3, [64, 64, 64])
        self.sa3 = PointNetSetAbstraction(350, 0.6, 32, 64+3, [128, 128, 128])
        self.sa4 = PointNetSetAbstraction(100, 0.9, 32, 128+3, [256, 256, 256])

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz
        Coded_set = []

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        Coded_set.append([l0_xyz, l0_points])
        Coded_set.append([l1_xyz, l1_points])
        Coded_set.append([l2_xyz, l2_points])
        Coded_set.append([l3_xyz, l3_points])
        return l4_xyz, l4_points, Coded_set

class PcDecoding(nn.Module):
    def __init__(self):
        super(PcDecoding, self).__init__()
        self.fp4 = PointNetFeaturePropagation(256+128, [256, 256])
        self.fp3 = PointNetFeaturePropagation(320, [256, 256])
        self.fp2 = PointNetFeaturePropagation(288, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 128, 1)

    def forward(self, xyz):

        l3_points = self.fp4(xyz[3][0], xyz[4][0], xyz[3][1], xyz[4][1])
        l2_points = self.fp3(xyz[2][0], xyz[3][0], xyz[2][1], l3_points)
        l1_points = self.fp2(xyz[1][0], xyz[2][0], xyz[1][1], l2_points)
        l0_points = self.fp1(xyz[0][0], xyz[1][0], None, l1_points)

        l_points = l0_points.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(l_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x

