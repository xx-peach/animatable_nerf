import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import tpose_renderer
from lib.train import make_optimizer
from . import crit
from lib.utils.if_nerf import if_nerf_net_utils
from lib.utils.blend_utils import *


class NetworkWrapper(nn.Module):
    """ NetworkWrapper for The Second Stage of Training
    just to train a new `net.novel_pose_bw` for novel poses, no need to
    bother with render loss, namely we don't need to compute rgb+sigma,
    the only concern here is blend weights, 这里做的事情就是用第一阶段在 train
    frames 上训练好的 net.bw 来训练一个新的给 novel poses 用的 bw
    """
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        # 'lib/networks/bw_deform/tpose_nerf_network.py'
        self.net = net
        self.renderer = tpose_renderer.Renderer(self.net)   #? no use
        # L1 consistency loss between blend weight fields
        self.bw_crit = torch.nn.functional.smooth_l1_loss
        self.img2mse = lambda x, y: torch.mean((x - y)**2)  #? no use
        # fix the pre-trained neural blend weight field for training frames
        for param in self.net.parameters():
            param.requires_grad = False
        # only train novel_pose_bw for novel poses
        for param in self.net.novel_pose_bw.parameters():
            param.requires_grad = True

    def forward(self, batch):
        # sample in observation space
        wpts = get_sampling_points(batch['wbounds'])        # (batch_size, nrays*N_samples, 3)
        ppts = wpts_to_ppts(wpts, batch)                    # (batch_size, nrays*N_samples, 3)
        pbw0, tbw0 = ppts_to_tpose(self.net, ppts, batch)
        # sample in canonical space
        tpts = get_sampling_points(batch['tbounds'])        # (batch_size, nrays*N_samples, 3)
        pbw1, tbw1 = tpose_to_ppts(self.net, tpts, batch)
        ret = {'pbw0': pbw0}

        scalar_stats = {}
        loss = 0

        bw_loss0 = self.bw_crit(pbw0, tbw0)
        bw_loss1 = self.bw_crit(pbw1, tbw1)
        scalar_stats.update({'bw_loss0': bw_loss0, 'bw_loss1': bw_loss1})
        loss += bw_loss0 + bw_loss1

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats


def ppts_to_tpose(net, pose_pts, batch):
    # blend weights of points at i
    pbw = pts_sample_blend_weights(pose_pts, batch['pbw'], batch['pbounds'])
    init_pbw, pnorm = pbw[:, :24], pbw[:, 24]

    # neural blend weights of points at i
    pbw = net.novel_pose_bw(pose_pts, init_pbw, batch['bw_latent_index'])

    # transform points from i to i_0
    tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])

    # calculate neural blend weights of points at the tpose space
    init_tbw = pts_sample_blend_weights(tpose, batch['tbw'], batch['tbounds'])
    init_tbw, tnorm = init_tbw[:, :24], init_tbw[:, 24]
    ind = torch.zeros_like(batch['bw_latent_index'])
    tbw = net.calculate_neural_blend_weights(tpose, init_tbw, ind)

    alpha = net.tpose_human.calculate_alpha(tpose)

    inside = tpose > batch['tbounds'][:, :1]
    inside = inside * (tpose < batch['tbounds'][:, 1:])
    inside = torch.sum(inside, dim=2) == 3
    # inside = inside * (tnorm < cfg.norm_th)
    inside = inside * (pnorm < cfg.norm_th)
    outside = ~inside
    alpha = alpha[:, 0]
    alpha[outside] = 0

    alpha_ind = alpha.detach() > cfg.train_th
    max_ind = torch.argmax(alpha, dim=1)
    alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
    pbw = pbw.transpose(1, 2)[alpha_ind]
    tbw = tbw.transpose(1, 2)[alpha_ind]

    return pbw, tbw


def tpose_to_ppts(net, tpose, batch):
    # calculate neural blend weights of points at the tpose space
    ind = torch.zeros_like(batch['bw_latent_index'])
    init_tbw = pts_sample_blend_weights(tpose, batch['tbw'], batch['tbounds'])
    init_tbw, tnorm = init_tbw[:, :24], init_tbw[:, 24]
    ind = torch.zeros_like(batch['bw_latent_index'])
    tbw = net.calculate_neural_blend_weights(tpose, init_tbw, ind)

    alpha = net.tpose_human.calculate_alpha(tpose)

    pose_pts = tpose_points_to_pose_points(tpose, tbw, batch['A'])

    # blend weights of points at i
    pbw = pts_sample_blend_weights(pose_pts, batch['pbw'], batch['pbounds'])
    init_pbw, pnorm = pbw[:, :24], pbw[:, 24]

    # neural blend weights of points at i
    pbw = net.novel_pose_bw(pose_pts, init_pbw, batch['bw_latent_index'])

    alpha = alpha[:, 0]

    alpha_ind = alpha.detach() > cfg.train_th
    max_ind = torch.argmax(alpha, dim=1)
    alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
    pbw = pbw.transpose(1, 2)[alpha_ind]
    tbw = tbw.transpose(1, 2)[alpha_ind]

    return pbw, tbw


def get_sampling_points(bounds):
    """ Sample nrays*N_samples Points inside the 3d Bounding Box
        随便采样就行, 我们只要将其转化到 T-pose 然后和训练好的 w^can 算 loss
    Arguments:
        bounds - (batch_size, 2, 3), bounding points of 6890 vertices
    Returns:
        pts - (batch_size, N_samples, 3), N_samples=1024*64 sampled points in bounds' coordinates
    """
    sh = bounds.shape       # (batch_size, 2, 3)
    min_xyz = bounds[:, 0]  # (batch_size, 3)
    max_xyz = bounds[:, 1]  # (batch_size, 3)
    N_samples = 1024 * 64
    x_vals = torch.rand([sh[0], N_samples])     # (batch_size, N_samples)
    y_vals = torch.rand([sh[0], N_samples])     # (batch_size, N_samples)
    z_vals = torch.rand([sh[0], N_samples])     # (batch_size, N_samples)
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)     # (batch_size, N_samples, 3)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts


def wpts_to_ppts(pts, batch):
    """Transform Points from the World Space to the Pose Space"""
    Th = batch['Th']
    pts = pts - Th
    R = batch['R']
    sh = pts.shape
    pts = torch.matmul(pts.view(sh[0], -1, 3), R)
    return pts
