import torch
import torch.nn.functional as F
import numpy as np


def world_points_to_pose_points(wpts, Rh, Th):
    """ Transfer Sampled Point from World to SMPL Coordinate
    Arguments:
        wpts - (1, batch_size*chunk*N_samples, 3), sampled points in world coordinate
        Rh   - (batch_size, 3, 3), smpl root joint's rotation matrix, smpl2world
        Th   - (batch_size, 1, 3), smpl root joint's translation matrix, smpl2world
    Returns:
        pts - (batch_size, batch_size*chunk*N_samples, 3), sampled points in smpl coordinate
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def world_dirs_to_pose_dirs(wdirs, Rh):
    """ Transfer Each Ray's Direction from World to SMPL Coordinate
    Arguments:
        wdirs - (1, batch_size*chunk*N_samples, 3), rays directions in world coordinate
        Rh    - (batch_size, 3, 3), smpl root joint's rotation matrix, smpl2world
    Returns:
        pts - (batch_size, batch_size*chunk*N_samples, 3), rays directions in smpl coordinate
    """
    pts = torch.matmul(wdirs, Rh)
    return pts


def pose_points_to_world_points(ppts, Rh, Th):
    """
    ppts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(ppts, Rh.transpose(1, 2)) + Th
    return pts


def pose_points_to_tpose_points(ppts, bw, A):
    """ Linear Blend Skinning, Transform Points from the SMPL Space to the T-pose(Canonical) Space
    Arguments:
        ppts - (batch_size, n', 3), sampled points in smpl coordinate
        bw   - (batch_size, 24, n'), final(neural + initial) blend weight for each sampled point
        A    - (24, 4, 4), G(poses, j_rel) * G(zero_pose, j)^{-1} of current frames in this batch
    Returns:
        pts - (batch_size, n', 3), sampled points' corresponding points in canonical space(T-pose)
    """
    sh = ppts.shape                                     # (batch_size, n', 3)
    bw = bw.permute(0, 2, 1)                            # (batch_size, n', 24)
    # compute the sum of w(x) * G_k
    A = torch.bmm(bw, A.view(sh[0], 24, -1))            # (batch_size, n', 16) <- (batch_size, n', 24) * (batch_size, 24, 4, 4)
    A = A.view(sh[0], -1, 4, 4)                         # (batch_size, n', 4, 4)
    # split (R|Th)^{-1}*ppts into ppts-Th and R^{-1}*(ppts-Th)
    pts = ppts - A[..., :3, 3]                          # (batch_size, n', 3)
    R_inv = torch.inverse(A[..., :3, :3])               # (batch_size, n', 3, 3)
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)     # (batch_size, n', 3)
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw, A):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * ddirs[:, :, None], dim=3)
    return pts


def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw, A):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * ddirs[:, :, None], dim=3)
    return pts


def grid_sample_blend_weights(grid_coords, bw):
    # the blend weight is indexed by xyz
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]
    return bw


def pts_sample_blend_weights(pts, bw, bounds):
    """ Sample Blend Weights for Sampled Points
    Arguments:
        pts    - (batch_size, n', 3), sampled points in smpl coordinate
        bw     - (batch_size, D, H, W, 25), input blend weight
        bounds - (batch_size, 2, 3), batch vertice's bounding points in smpl coordinate
    Returns:
        bw - (batch_size, 25, n'), interpolated blend weight for each sampled points
    """
    pts = pts.clone()

    # compute grid coordinate of all sampled points inside the smpl vertice volume
    min_xyz = bounds[:, 0]                          # (batch_size, 3)
    max_xyz = bounds[:, 1]                          # (batch_size, 3)
    bounds = max_xyz[:, None] - min_xyz[:, None]    # (batch_size, 1, 3)
    grid_coords = (pts - min_xyz[:, None]) / bounds # (batch_size, n', 3)
    # transfer the grid coordinates from range [0, 1] to [-1, 1]
    grid_coords = grid_coords * 2 - 1               # (batch_size, n', 3)

    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]       # (batch_size, n', 3)

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)                  # (batch_size, 25, D, H, W)
    grid_coords = grid_coords[:, None, None]        # (batch_size, 1, 1, n', 3)
    bw = F.grid_sample(bw,                          # (N, C, Din, Hin, Win) <- (bs, 25, D, H, W)
                       grid_coords,                 # (N, Dot, Hot, Wot, 3) <- (bs, 1, 1, n', 3)
                       padding_mode='border',
                       align_corners=True)          # (N, C, Dot, Hot, Wot) <- (bs, 25, 1, 1, n')
    bw = bw[:, :, 0, 0]                             # (batch_size, 25, n')
    return bw


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x N_samples x 24 x 3
    bw: batch_size x 24 x 64 x 64 x 64
    """
    bws = []
    for i in range(24):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = F.grid_sample(bw[:, i:i + 1],
                            nf_grid_coords_,
                            padding_mode='border',
                            align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = torch.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, N_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = torch.rand([sh[0], N_samples])
    y_vals = torch.rand([sh[0], N_samples])
    z_vals = torch.rand([sh[0], N_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts
