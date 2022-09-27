import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
from lib.utils import sample_utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # a standard NeRF model architecture
        self.tpose_human = TPoseHuman()
        # latent codes for different frames or poses more specifically
        self.bw_latent = nn.Embedding(cfg.num_train_frame + 1, 128)
        # use ReLU as activation layer
        self.actvn = nn.ReLU()
        # network architecture for neural blend weight calculation
        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

        if cfg.aninerf_animation:
            self.novel_pose_bw = BackwardBlendWeight()

            if 'init_aninerf' in cfg:
                net_utils.load_network(self,
                                       'data/trained_model/deform/' +
                                       cfg.init_aninerf,
                                       strict=False)

    def get_bw_feature(self, pts, ind):
        """ 
        Arguments:
            pose_pts - (batch_size, n', 3), sampled points in smpl coordinate
            ind      - (batch_size,), view index \in [0, self.num_cams-1]
        Returns:
            features - (batch_size, 128+k, n') -> (batch_size, 191, chunk*N_samples)
        """
        pts = embedder.xyz_embedder(pts)    # (batch_size, n', 63), positional embedding for sampled points in smpl coordinate
        pts = pts.transpose(1, 2)           # (batch_size, 63, n')
        latent = self.bw_latent(ind)        # (batch_size, 128)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))   # (batch_size, 128, n')
        features = torch.cat((pts, latent), dim=1)                      # (batch_size, 191, n')
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        """ Compute the Blend Weights(Neural + Initial) for each Sampled Point in SMPL Coordinate
        Arguments:
            pose_pts     - (batch_size, n', 3), sampled points in smpl coordinate
            smpl_bw      - (batch_size, 24, n'), initial smpl blend weight for each sampled point
            latent_index - (batch_size, 1), frame index for train;;; and num_train_frame(new) for test????
        Returns:
            bw - (batch_size, 24, n'), final(neural + initial) blend weight for each sampled point
        """
        # concatenate the positional-embedded xyz and latent index to form the feature as input for self.bw_linears
        features = self.get_bw_feature(pose_pts, latent_index)  # (batch_size, 191, n')
        net = features
        # go through the self.bw_linears to get neural blend weights
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))           # (batch_size, 256, n')
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)                                    # (batch_size, 24, n')
        # add neural blend weight and initial smpl weight together
        bw = torch.log(smpl_bw + 1e-9) + bw                     # (batch_size, 24, n')
        # go through softmax to get the final weight
        bw = F.softmax(bw, dim=1)                               # (batch_size, 24, n')
        return bw

    def pose_points_to_tpose_points(self, pose_pts, batch):
        """ Transfer Sampled Points in SMPL Coordinate to Corresponding T-pose Points(Canonical Space)
        Arguments:
            pose_pts - (batch_size, n', 3), sampled points in smpl coordinate
            batch    - a dict() with ['pbw'](initial bw), ['pbounds'](compute grid coordinates), and ['(bw)latent_index'] we want
        Returns:
            tpose - (batch_size, n', 3), sampled points' corresponding points in canonical space(T-pose)
            pbw   - (batch_size, 24, n'), final(neural + initial) blend weight for each sampled point
        """
        # initial blend weights of points at i
        init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'], batch['pbounds'])   # (batch_size, 25, n')
        init_pbw = init_pbw[:, :24]                                                     # (batch_size, 24, n')

        # neural blend weights of points at i, with initial blend weights added
        if cfg.test_novel_pose:
            pbw = self.novel_pose_bw(pose_pts, init_pbw, batch['bw_latent_index'])      # (batch_size, 24, n')
        else:
            pbw = self.calculate_neural_blend_weights(pose_pts, init_pbw, batch['latent_index'] + 1)

        # transform points from i in smpl to i_0 in canonical(T-pose), linear blend skinning
        tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])
        return tpose, pbw

    # NOTE: this part should actually have been deprecated...
    # we leave this here for reproducability, in the extended version, we implmented a better aninerf pipeline (same core idea as the paper)
    # thus some of the old config files or code could not run as expected especially when outside the core training loop
    def calculate_alpha(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        pnorm = init_pbw[:, 24]
        norm_th = 0.1
        pind = pnorm < norm_th
        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
        pose_pts = pose_pts[pind][None]

        # transform points from the pose space to the tpose space
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        # calculate neural blend weights of points at the tpose space
        init_tbw = pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)

        alpha = self.tpose_human.calculate_alpha(tpose)
        alpha = alpha[0, 0]

        n_batch, n_point = wpts.shape[:2]
        full_alpha = torch.zeros([n_point]).to(wpts)
        full_alpha[pind[0]] = alpha

        return full_alpha

    get_alpha = calculate_alpha

    def forward(self, wpts, viewdir, dists, batch):
        ###########################################################
        # transform points from the world space to the pose space #
        ###########################################################
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])   # (batch_size, chunk*N_samples, 3)

        ##########################################################################
        # filter those points with larger initial blend weights than cfg.norm_th #
        ##########################################################################
        with torch.no_grad():
            init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'], batch['pbounds'])   # (batch_size, 25, chunk*N_samples)
            pnorm = init_pbw[:, -1]             # (batch_size, chunk*N_samples)
            norm_th = cfg.norm_th               # default 0.05
            pind = pnorm < norm_th              # (batch_size, n')
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]     # (batch_size, n', 3)
            viewdir = viewdir[pind[0]]          # (batch_size, n', 3)
            dists = dists[pind[0]]              # (n')

        #################################################################################################
        # transform points from smpl to the tpose space, and returns tpose points + final blend weight  #
        #* tpose: (batch_size, n', 3), sampled points' corresponding points in canonical space(T-pose)  #
        #* pbw  : (batch_size, 24, n'), final(neural + initial) blend weight for each sampled point     #
        #################################################################################################
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        ###############################################################
        # calculate neural blend weights of points at the tpose space #
        ###############################################################
        # calculate initial blend weights for each points by interpolation
        init_tbw = pts_sample_blend_weights(tpose, batch['tbw'], batch['tbounds'])  # (batch_size, 25, n')
        init_tbw = init_tbw[:, :24]                                                 # (batch_size, 24, n')
        # calculate blend weights by self.bw_linears and add them to initial blend weights to get final `tbw`
        ind = torch.zeros_like(batch['latent_index'])                               # (batch_size, 1)
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)             # (batch_size, 24, n')

        ######################################################################
        # construct a NeRF in canonical space, namely the fixed T-pose scene #
        ######################################################################
        viewdir = viewdir[None]
        ind = batch['latent_index']
        alpha, rgb = self.tpose_human.calculate_alpha_rgb(tpose, viewdir, ind)      # (batch_size, 1, n'), (batch_size, 3, n')

        #############################################################################################
        # further filter, 将由 neural blend weights 转化到 canonical space 的超出原本 T-pose 范围的点剔除 #
        #############################################################################################
        inside = tpose > batch['tbounds'][:, :1]
        inside = inside * (tpose < batch['tbounds'][:, 1:])
        outside = torch.sum(inside, dim=2) != 3
        alpha = alpha[:, 0]         # (batch_size, n')
        alpha[outside] = 0          # (batch_size, n')

        alpha_ind = alpha.detach() > cfg.train_th               # (batch_size, n')
        max_ind = torch.argmax(alpha, dim=1)                    # (batch_size,)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True  # (batch_size, n')
        pbw = pbw.transpose(1, 2)[alpha_ind][None]              # (1, batch_size, 24)
        tbw = tbw.transpose(1, 2)[alpha_ind][None]              # (1, batch_size, 24)

        ###################################################################################
        # raw2output: compute rgb color and transparency using volume density + distances #
        ###################################################################################
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
        rgb = torch.sigmoid(rgb[0])                 # (3, n')
        alpha = raw2alpha(alpha[0], dists)          # (n')
        raw = torch.cat((rgb, alpha[None]), dim=0)  # (4, n')
        raw = raw.transpose(0, 1)                   # (n', 4)

        ################################################################################################
        # fill those points who have been filtered(initial blend filter + T-pose bbox filter) out by 0 #
        ################################################################################################
        n_batch, n_point = wpts.shape[:2]           # (batch_size, chunk*N_samples)
        raw_full = torch.zeros([n_batch, n_point, 4], dtype=wpts.dtype, device=wpts.device) # (batch_size, chunk*N_samples, 4)
        raw_full[pind] = raw                        # (batch_size, chunk*N_samples, 4)

        ret = {'pbw': pbw, 'tbw': tbw, 'raw': raw_full}
        return ret


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.nf_latent = nn.Embedding(cfg.num_train_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 63
        D = 8
        W = 256
        self.skips = [4]
        self.pts_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.alpha_fc = nn.Conv1d(W, 1, 1)

        self.feature_fc = nn.Conv1d(W, W, 1)
        self.latent_fc = nn.Conv1d(384, W, 1)
        self.view_fc = nn.Conv1d(283, W // 2, 1)
        self.rgb_fc = nn.Conv1d(W // 2, 3, 1)

    def calculate_alpha(self, nf_pts):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)
        return alpha

    def calculate_alpha_rgb(self, nf_pts, viewdir, ind):
        nf_pts = embedder.xyz_embedder(nf_pts)
        input_pts = nf_pts.transpose(1, 2)
        net = input_pts
        for i, l in enumerate(self.pts_linears):
            net = self.actvn(self.pts_linears[i](net))
            if i in self.skips:
                net = torch.cat((input_pts, net), dim=1)
        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.nf_latent(ind)
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        features = torch.cat((features, viewdir), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        return alpha, rgb


class BackwardBlendWeight(nn.Module):
    def __init__(self):
        super(BackwardBlendWeight, self).__init__()

        self.bw_latent = nn.Embedding(cfg.num_eval_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, ppts, smpl_bw, latent_index):
        latents = self.bw_latent
        features = self.get_point_feature(ppts, latent_index, latents)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw
