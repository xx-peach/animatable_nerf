import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
import os
from lib.utils import sample_utils


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.resd_latent = nn.Embedding(cfg.num_latent_code, 128)

        self.actvn = nn.ReLU()

        input_ch = 135
        D = 8
        W = 256
        self.skips = [4]
        self.resd_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.resd_fc = nn.Conv1d(W, 3, 1)
        self.resd_fc.bias.data.fill_(0)

        if cfg.get('init_sdf', False):
            init_path = os.path.join('data/trained_model', cfg.task,
                                     cfg.init_sdf)
            net_utils.load_network(self,
                                   init_path,
                                   only=['tpose_human.sdf_network'])

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_residual_deformation(self, tpose, batch):
        """ Neural Displacement Field in Canonical Space
        Arguments:
            tpose - (batch_size, n', 3), sample points after transferred to big-pose(first to T-pose, and then to big-pose)
            batch - a dict(), we need its ['pose'] to predict residual displacement in big-pose
        Returns:
            resd - (batch_size, n', 3), residual displacement \Delta x in big-pose
        """
        # perform positional encoding to big-pose sample points
        pts = embedder.xyz_embedder(tpose)          # (batch_size, n', 63)
        pts = pts.transpose(1, 2)                   # (batch_size, 63, n')
        # fetch current frame's human pose (72,) from batch
        latent = batch['poses']                     # (batch_size, 72)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))   # (batch_size, 72, n')
        features = torch.cat((pts, latent), dim=1)  # (batch_size, 72+63, n')

        net = features
        for i, l in enumerate(self.resd_linears):
            net = self.actvn(self.resd_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        resd = self.resd_fc(net)                    # (batch_size, 3, n')
        resd = resd.transpose(1, 2)                 # (batch_size, n', 3)
        resd = 0.05 * torch.tanh(resd)              # (batch_size, n', 3)
        return resd

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """ Transform Points from the Pose Space to the T-pose Space
        Arguments:
            pose_pts  - (batch_size, n', 3), filtered sampled points in smpl coordinate
            pose_dirs - (batch_size, n', 3), filtered sampled points' view directions in smpl coordinate
            batch     - a dict() with ['A'] and ['big_A'](大字型 human pose) we want
        Returns:
            tpose        - (batch_size, n', 3), final accurate sample points coordinates in big-pose
            tpose_dirs   - (batch_size, n', 3), corresponding view directions transferred to big-pose
            init_bigpose - (batch_size, n', 3), coarse sample points coordinates in big-pose
            resd         - (batch_size, n', 3), residual displacement \Delta x in big-pose
        """
        # initial blend weights of points at i, of shape (batch_size, chunk*N_samples, 24)
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1)          # (batch_size, 24, chunk*N_samples)

        # transform points from i to i_0(current pose to T pose)
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])         # (batch_size, n', 3)
        # transform points from T-pose just computed to big pose(预定义的大字型 pose)
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw, batch['big_A']) # (batch_size, n', 3)
        # predict residual displacement \Delta x of each sample point who has been transferred into big-pose
        resd = self.calculate_residual_deformation(init_bigpose, batch)
        # final accurate sample points coordinates in big-pose
        tpose = init_bigpose + resd                                                 # (batch_size, n', 3)

        # transfer the view directions to T-pose and big-pose is specified
        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw, batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw, batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd

    def calculate_bigpose_smpl_bw(self, bigpose, input_bw):
        smpl_bw = pts_sample_blend_weights(bigpose, input_bw['tbw'],
                                           input_bw['tbounds'])
        return smpl_bw

    def calculate_wpts_sdf(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(
            pose_pts, None, batch)
        tpose = tpose[0]
        sdf = self.tpose_human.sdf_network(tpose, batch)[:, :1]

        return sdf

    def wpts_gradient(self, wpts, batch):
        wpts.requires_grad_(True)
        with torch.enable_grad():
            sdf = self.calculate_wpts_sdf(wpts, batch)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients

    def gradient_of_deformed_sdf(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            resd = self.calculate_residual_deformation(x, batch)
            tpose = x + resd
            tpose = tpose[0]
            y = self.tpose_human.sdf_network(tpose, batch)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients, y[None]

    def forward(self, wpts, viewdir, dists, batch):
        ###########################################################
        # transform points from the world space to the pose space #
        ###########################################################
        wpts = wpts[None]               # (1, batch_size*chunk*N_samples, 3), #! 只是因为 batch_size==1 才没错
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])   # (batch_size, batch_size*chunk*N_samples, 3) <- (batch_size, chunk*N_samples, 3)
        viewdir = viewdir[None]         # (1, batch_size*chunk*N_samples, 3), #! 只是因为 batch_size==1 才没错
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])                # (batch_size, batch_size*chunk*N_samples, 3) <- (batch_size, chunk*N_samples, 3)

        ##########################################################################
        # filter those points with larger initial blend weights than cfg.norm_th #
        ##########################################################################
        with torch.no_grad():
            # pnorm: (batch_size, chunk*N_samples, 1), each sample point 与 k 个 nearest neighbors 之间的加权距离
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = 0.1
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]     # (batch_size, n', 3)
            viewdir = viewdir[pind][None]       # (batch_size, n', 3), world view directions
            pose_dirs = pose_dirs[pind][None]   # (batch_size, n', 3), smpl view directions

        #####################################################################
        # transform points from the pose space to the tpose(big-pose) space #
        #####################################################################
        tpose, tpose_dirs, init_bigpose, resd = self.pose_points_to_tpose_points(pose_pts, pose_dirs, batch)
        tpose = tpose[0]                # (n', 3), 因为 batch_size = 1
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]     # (n', 3), 因为 batch_size = 1
        else:
            viewdir = viewdir[0]        # (n', 3), 因为 batch_size = 1
        
        #################################################################################################
        # construct a sdf network and rendering network in big-pose, a pre-defined fixed big-pose scene #
        #################################################################################################
        ret = self.tpose_human(tpose, viewdir, dists, batch)        # (n', 4)

        ind = ret['sdf'][:, 0].detach().abs() < 0.02
        init_bigpose = init_bigpose[0][ind][None].detach().clone()

        if ret['raw'].requires_grad and ind.sum() != 0:
            observed_gradients, _ = self.gradient_of_deformed_sdf(init_bigpose, batch)
            ret.update({'observed_gradients': observed_gradients})

        ###############################################################################################
        # further filter, 将由 neural blend weights 转化到 canonical space 的超出原本 big-pose 范围的点剔除 #
        ###############################################################################################
        tbounds = batch['tbounds'][0]
        tbounds[0] -= 0.05
        tbounds[1] += 0.05
        inside = tpose > tbounds[:1]
        inside = inside * (tpose < tbounds[1:])
        outside = torch.sum(inside, dim=1) != 3
        ret['raw'][outside] = 0

        ##################################################################################################
        # fill those points who have been filtered(initial blend filter + big-pose bbox filter) out by 0 #
        ##################################################################################################
        n_batch, n_point = wpts.shape[:2]
        raw = torch.zeros([n_batch, n_point, 4]).to(wpts)       # (batch_size, chunk*N_samples, 4)
        raw[pind] = ret['raw']
        sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)   # (batch_size, chunk*N_samples, 1)
        sdf[pind] = ret['sdf']
        ret.update({'raw': raw, 'sdf': sdf, 'resd': resd})      # (batch_size, chunk*N_samples, 3)

        ret.update({'gradients': ret['gradients'][None]})       # (batch_size, n', 3)

        return ret

    def get_sdf(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = 0.1
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]

        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        pbw = pbw.permute(0, 2, 1)

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])
        resd = self.calculate_residual_deformation(init_bigpose, batch)
        tpose = init_bigpose + resd
        tpose = tpose[0]

        sdf_nn_output = self.tpose_human.sdf_network(tpose, batch)
        sdf = sdf_nn_output[:, 0]

        n_batch, n_point = wpts.shape[:2]
        sdf_full = 10 * torch.ones([n_batch, n_point]).to(wpts)
        sdf_full[pind] = sdf
        sdf = sdf_full.view(-1, 1)

        return sdf


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.sdf_network = SDFNetwork()
        self.beta_network = BetaNetwork()
        self.color_network = ColorNetwork()

    def sdf_to_alpha(self, sdf, beta):
        x = -sdf

        # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
        ind0 = x <= 0
        val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

        # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
        ind1 = x > 0
        val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

        val = torch.zeros_like(sdf)
        val[ind0] = val0
        val[ind1] = val1

        return val

    def forward(self, wpts, viewdir, dists, batch):
        """ Compute Sdf, Normal, RGB Color and Alpha
        Arguments:
            wpts    - (n', 3), filtered sample points in big-pose
            viewdir - (n', 3), corresponding view direction of each sample point
            dists   - (batch_size*chunk*N_samples,), distance between sample points along the same ray
                      #? 这个 dist 就没有用到, 按照 alpha 的计算公式是要用的, 但是下面的代码里改成了 * 0.05
            batch   - a dict(), with original ['latent_index']
        Returns:
            ret {
                raw       - (n', 4), rgb color + computed alpha
                sdf       - (n', 1), signed distance of each point
                gradients - (n', 3), normal vector of each point
            }
        """
        ####################################
        # calculate sdf and feature vector #
        ####################################
        wpts.requires_grad_()
        with torch.enable_grad():
            sdf_nn_output = self.sdf_network(wpts, batch)
            sdf = sdf_nn_output[:, :1]                  # (n', 1)
        feature_vector = sdf_nn_output[:, 1:]           # (n', 256)

        ####################
        # calculate normal #
        ####################
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        # gradients = self.sdf_network.gradient(wpts, batch)[:, 0]

        ###################################
        # calculate alpha(volume density) #
        ###################################
        wpts = wpts.detach()                            # (n', 3), 其实对下面没有作用
        beta = self.beta_network(wpts).clamp(1e-9, 1e6) # fetch current beta, 传进去的 wpts 没有用
        alpha = self.sdf_to_alpha(sdf, beta)            # (n', 1), volume density 其实是 \sigma
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)           # (n',), 由 sigma 算得的 alpha

        ###################
        # calculate color #
        ###################
        ind = batch['latent_index']
        rgb = self.color_network(wpts, gradients, viewdir, feature_vector, ind)

        ##############################################
        # return rgb color, alpha, sdf and gradients #
        ##############################################
        raw = torch.cat((rgb, alpha[:, None]), dim=1)   # (n', 4)
        ret = {'raw': raw, 'sdf': sdf, 'gradients': gradients}

        return ret


class SDFNetwork(nn.Module):
    def __init__(self):
        super(SDFNetwork, self).__init__()

        d_in = 3            # input channel, default = 3
        d_out = 257         # output channel, default = 1+256
        d_hidden = 256      # hidden layer channels, default = 256
        n_layers = 8        # number of layers, default = 256

        # each layer's channel from input to output
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        # instantiate the embedder for input sample points
        self.embed_fn_fine = None
        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        skip_in = [4]       # similar to NeRF, resblock index, default = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, batch):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, batch):
        return self.forward(x, batch)[:, :1]

    def gradient(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x, batch)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class BetaNetwork(nn.Module):
    def __init__(self):
        super(BetaNetwork, self).__init__()
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        beta = self.beta
        # beta = torch.exp(self.beta).to(x)
        return beta


class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)

        d_feature = 256
        mode = 'idr'
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 4
        squeeze_out = True

        if not cfg.get('color_with_viewdir', True):
            mode = 'no_view_dir'

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden
                                     for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if self.mode != 'no_view_dir':
            multires_view = 4
            if multires_view > 0:
                embedview_fn, input_ch = embedder.get_embedder(multires_view)
                self.embedview_fn = embedview_fn
                dims[0] += (input_ch - 3)
        else:
            dims[0] = dims[0] - 3

        self.num_layers = len(dims)

        self.lin0 = nn.Linear(dims[0], d_hidden)
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden + 128, d_hidden)
        self.lin4 = nn.Linear(d_hidden, d_out)

        weight_norm = True
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors,
                latent_index):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat(
                [points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors],
                                        dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors],
                                        dim=-1)

        x = rendering_input

        net = self.relu(self.lin0(x))
        net = self.relu(self.lin1(net))
        net = self.relu(self.lin2(net))

        latent = self.color_latent(latent_index)
        latent = latent.expand(net.size(0), latent.size(1))
        features = torch.cat((net, latent), dim=1)

        net = self.relu(self.lin3(features))
        x = self.lin4(net)

        if self.squeeze_out:
            x = torch.sigmoid(x)

        return x
