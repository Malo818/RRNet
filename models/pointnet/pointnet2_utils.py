import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


# def square_distance(src, dst):
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return dist

# def query_ball_point(radius, nsample, xyz, new_xyz):
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])       # [B, S, N]
#     sqrdists = square_distance(new_xyz, xyz)                                                      #  [B, S, N]
#     group_idx[sqrdists > radius ** 2] = N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#     return group_idx

def square_distance_batch(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    if M%N >=30:
        dist_idx = torch.zeros((B, N, 30*((M//N)+1)), device=src.device)
        dist_val = torch.zeros((B, N, 30*((M//N)+1)), device=src.device)
        j = 0
        for i in range(0, M, N):
            end = min(i + N, M)
            dst_part = dst[:, i:end, :]
            dist_part = -2 * torch.matmul(src, dst_part.permute(0, 2, 1))
            dist_part += torch.sum(src ** 2, -1).view(B, N, 1)
            dist_part += torch.sum(dst_part ** 2, -1).view(B, 1, end - i)

            group_idx = dist_part.sort(dim=-1)[1][:, :, :30] + i
            group_val = dist_part.sort(dim=-1)[0][:, :, :30]

            dist_idx[:, :, j:j+30] = group_idx
            dist_val[:, :, j:j+30] = group_val
            j +=30
    else:
        dist_idx = torch.zeros((B, N, 30*(M//N)), device=src.device)
        dist_val = torch.zeros((B, N, 30*(M//N)), device=src.device)

        j = 0
        for i in range(0, M-(M%N), N):
            end = min(i + N, M)
            dst_part = dst[:, i:end, :]
            dist_part = -2 * torch.matmul(src, dst_part.permute(0, 2, 1))
            dist_part += torch.sum(src ** 2, -1).view(B, N, 1)
            dist_part += torch.sum(dst_part ** 2, -1).view(B, 1, end - i)

            group_idx = dist_part.sort(dim=-1)[1][:, :, :30] + i
            group_val = dist_part.sort(dim=-1)[0][:, :, :30]

            dist_idx[:, :, j:j+30] = group_idx
            dist_val[:, :, j:j+30] = group_val
            j +=50
    return dist_val, dist_idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists, idx = square_distance_batch(new_xyz, xyz)          # Point clouds are partitioned extensively to prevent memory overflow.

    sorted_sqrdists, sorted_indices = torch.sort(sqrdists, dim=-1)
    mask = sorted_sqrdists > radius ** 2
    first = sorted_indices[:, :, 0].unsqueeze(-1).repeat(1, 1, sorted_sqrdists.size(2))
    sorted_indices[mask] = first[mask]
    group = sorted_indices[:, :, :32]
    result = torch.gather(idx, dim=-1, index=group)
    return result.long()


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):

    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)                         # [B, S]
    new_xyz = index_points(xyz, fps_idx)                                 # [B, S, 3]

    idx = query_ball_point(radius, nsample, xyz, new_xyz)               #  [B, S, K]
    grouped_xyz = index_points(xyz, idx)                               # [B, S, K, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)          #  [B, S, K, 3]

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):

        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, 3+D, K,S]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance_batch1(xyz1, xyz2)       # [mb,N,S]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)
        return new_points

def square_distance_batch1(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist_val = torch.zeros((B, N, 3), device=src.device)
    dist_idx = torch.zeros((B, N, 3), device=src.device)

    for i in range(0, N, M):
        end = min(i + M, N)
        src_part = src[:, i:end, :]
        dist_part = -2 * torch.matmul(src_part, dst.permute(0, 2, 1))
        dist_part += torch.sum(src_part ** 2, -1).view(B, end - i, 1)
        dist_part += torch.sum(dst ** 2, -1).view(B, 1, M)

        dist_val[:, i:end, :] = dist_part.sort(dim=-1)[0][:, :, :3]
        dist_idx[:, i:end, :] = dist_part.sort(dim=-1)[1][:, :, :3]
    return dist_val, dist_idx.long()
