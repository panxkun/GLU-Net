import torch
import torch.nn.functional as F
import numpy as np


def grid_gen(batch_size,ht,wd):
    x = torch.arange(0, wd, 1)
    y = torch.arange(0, ht, 1)
    xx, yy = torch.meshgrid(x, y)
    coord_flow = torch.stack([xx, yy], dim = 2)
    return coord_flow.permute([2, 1, 0]).repeat([batch_size, 1, 1, 1])


def image_pair_symmetric_distance(x, y, F, width, height):
    # x, y: (B * H * W) x 3
    # F: B x 3 x 3
    # y^TFx * ( 1 / ((Fx)_1^2 + (Fx)_2^2) + 1 / ((FTy)_1^2 + (FTy)_2^2))

    epi_dis = torch.sum(torch.bmm(y, F) * x, dim=2).reshape(-1, width * height)
    epi_line1 = torch.bmm(x, F.permute([0, 2, 1]))
    epi_line2 = torch.bmm(y, F)

    epi_dis = epi_dis * (
            1.0 / torch.clamp(torch.norm(epi_line1[:, :, :2], dim=2, p=2), min=1e-06) +
            1.0 / torch.clamp(torch.norm(epi_line2[:, :, :2], dim=2, p=2), min=1e-06)
    )

    return epi_dis


def geometry_loss(flow_est, F):

    batch_size, _, height, width = flow_est.shape
    coord_a = grid_gen(batch_size, height, width).float().cuda()
    homo_coord_a = torch.cat((coord_a, torch.ones([batch_size, 1, height, width]).float().cuda()), dim=1)
    shuffle_homo_coord_a = homo_coord_a.permute([0, 2, 3, 1]).reshape([batch_size, -1, 3])

    homo_coord_b = torch.cat(
        (coord_a + flow_est, torch.ones([batch_size, 1, height, width]).float().cuda()),
        dim=1
    )
    shuffle_homo_coord_b = homo_coord_b.permute([0, 2, 3, 1]).reshape([batch_size, -1, 3])

    sym_distance = image_pair_symmetric_distance(shuffle_homo_coord_a, shuffle_homo_coord_b, F, width, height)
    sym_dis_loss = torch.mean(torch.abs(sym_distance))

    return sym_dis_loss