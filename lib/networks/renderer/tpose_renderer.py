import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        # 'lib/networks/bw_deform/tpose_nerf_network.py'
        self.net = net

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        """ Sample N_sample Points from each Ray
        Arguments:
            ray_o - (batch_size, chunk, 3), rays origin in the world coordinates
            ray_d - (batch_size, chunk, 3), rays direction in world coordinates
            near  - (batch_size, chunk), near distance of each ray in world coordinates
            far   - (batch_size, chunk), far distance of each ray in world coordinates
        Returns:
            pts    - (batch_size, chunk, N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            z_vals - (batch_size, chunk, N_samples), N_sample points' z value along their corresponding ray direction
        """
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)       # (N_samples,)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals  # (batch_size, chunk, N_samples)

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]     # (batch_size, chunk, N_samples, 3)
        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """ Prepare Input Sampled-Points and View Directions for NeRF
        Arguments:
            wpts        - (batch_size, chunk, N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            viewdir     - (batch_size, chunk, 3), normalized rays direction for each ray in this batch
            z_vals      - (batch_size, chunk, N_samples), N_sample points' z value along the their corresponding ray direction
            raw_decoder - a lambda function that call the true Network.calculate_density_color()
        Returns:
            ret - {
                'pbw'  - (1, batch_size, 24), lbw 的 return, #? 还不知道为什么是这个 shape
                'tbw'  - (1, batch_size, 24), lbw 的 return, #? 还不知道为什么是这个 shape
                'raw'  - (batch_size, chunk*N_samples, 4), rgb + color
                'sdf'  - (batch_size, chunk*N_samples, 1), signed distance, anisdf 的 return
                'resd' - (batch_size, chunk*N_samples, 3), displacement field output, anisdf 的 return
                'gradient' - (batch_size, n', 3), normal vector, anisdf 的 return
            }
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]                             # batch_size, chunk, N_samples
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)                      # (batch_size*chunk*N_samples, 3)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()    # (batch_size, chunk, N_samples, 3)
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)                # (batch_size*chunk*N_samples, 3)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]          # (batch_size, chunk, N_samples-1)
        dists = torch.cat([dists, dists[..., -1:]], dim=2)  # (batch_size, chunk, N_samples)
        dists = dists.view(n_batch * n_pixel * n_sample)    # (batch_size*chunk*N_samples,)

        ret = raw_decoder(wpts, viewdir, dists)
        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        """ Given ray_o, ray_d, near, far, feature_volume, Go Through NeRF to Get Density and Color
        Arguments:
            ray_o - (batch_size, chunk, 3), rays origin in the world coordinates
            ray_d - (batch_size, chunk, 3), rays direction in world coordinates
            near  - (batch_size, chunk), near distance of each ray in world coordinates
            far   - (batch_size, chunk), far distance of each ray in world coordinates
            occ   - (batch_size, chunk), whether the chosen pixel >0 or not in the original mask
            batch - dict(), original input returned by dataloader
        Returns:
            rgb_map   - (batch_size*chunk, 3), RGB color of rays in this batch
            disp_map  - (batch_size*chunk,), disparity of rays in this batch
            acc_map   - (batch_size*chunk,), sum of weights along each ray
            weights   - (batch_size*chunk, N_samples),  weights assigned to each sampled color
            depth_map - (batch_size*chunk,), depth of rays in this batch
        """
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)   # (batch_size, chunk, N_samples, 3)
        n_batch, n_pixel, n_sample = wpts.shape[:3]                         # (batch_size, chunk, N_samples)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d                                                     # (batch_size, chunk, 3)
        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(wpts_val, viewdir_val, dists_val, batch)
        
        """ compute the color and density
        ret - {
            'pbw'  - (1, batch_size, 24), lbw 的 return, 这是给 train 用的, #? 还不知道为什么是这个 shape
            'tbw'  - (1, batch_size, 24), lbw 的 return, 这是给 train 用的, #? 还不知道为什么是这个 shape
            'sdf'  - (batch_size, chunk*N_samples, 1), signed distance, anisdf 的 return, 这是给 train 用的
            'resd' - (batch_size, chunk*N_samples, 3), displacement field output, anisdf 的 return, 这是给 train 用的
            'gradient' - (batch_size, n', 3), normal vector, anisdf 的 return, 这是给 train 用的
            'raw'  - (batch_size, chunk*N_samples, 4), rgb + color
        } """
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape       # (batch_size, chunk, N_samples)
        raw = ret['raw'].reshape(-1, n_sample, 4)       # (batch_size*chunk, N_samples, 4)
        z_vals = z_vals.view(-1, n_sample)              # (batch_size*chunk, N_samples)
        ray_d = ray_d.view(-1, 3)                       # (batch_size*chunk, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)    # (batch_size, chunk, 3)
        acc_map = acc_map.view(n_batch, n_pixel)        # (batch_size, chunk)
        depth_map = depth_map.view(n_batch, n_pixel)    # (batch_size, chunk)

        ret.update({
            'rgb_map': rgb_map,                 # (batch_size, chunk, 3)
            'acc_map': acc_map,                 # (batch_size, chunk)
            'depth_map': depth_map,             # (batch_size, chunk)
            'raw': raw.view(n_batch, -1, 4)     # (batch_size, chunk*N_samples, 4)
        })

        if 'pbw' in ret:
            pbw = ret['pbw'].view(n_batch, -1, 24)
            ret.update({'pbw': pbw})            # (batch_size, 1, 24)

        if 'tbw' in ret:
            tbw = ret['tbw'].view(n_batch, -1, 24)
            ret.update({'tbw': tbw})            # (batch_size, 1, 24)

        if 'sdf' in ret:
            # get pixels that outside the mask or no ray-geometry intersection
            sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
            min_sdf = sdf.min(dim=2)[0]
            free_sdf = min_sdf[occ == 0]
            free_label = torch.zeros_like(free_sdf)

            with torch.no_grad():
                intersection_mask, _ = get_intersection_mask(sdf, z_vals)
            ind = (intersection_mask == False) * (occ == 1)
            sdf = min_sdf[ind]
            label = torch.ones_like(sdf)

            sdf = torch.cat([sdf, free_sdf])
            label = torch.cat([label, free_label])
            ret.update({
                'msk_sdf': sdf.view(n_batch, -1),
                'msk_label': label.view(n_batch, -1)
            })

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def render(self, batch):
        # fetch needed data from current batch
        ray_o = batch['ray_o']      # (batch_size, nrays, 3), rays origin in world coordinates
        ray_d = batch['ray_d']      # (batch_size, nrays, 3), rays direction in world coordinates
        near = batch['near']        # (batch_size, nrays), near distance of each ray in world coordinates
        far = batch['far']          # (batch_size, nrays), far distance of each ray in world coordinates
        occ = batch['occupancy']    # (batch_size, nrays), whether the chosen pixel >0 or not in the original mask
        sh = ray_o.shape            # (batch_size, nrays, 3) --> sh

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]          # n_batch=batch_size, n_pixel=nrays
        chunk = 2048                                # avoid CUDA out of memory
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            occ_chunk = occ[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
