import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *
from . import tpose_renderer


class Renderer(tpose_renderer.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def prepare_inside_pts(self, pts, batch):
        """ Prepare Inside Points
        Arguments:
            pts   - (1, chunk, N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            batch - dict(), original input returned by dataloader
        Returns:
            inside - (1, chunk*N_samples), if 1 说明当前 render view 的这个 sample point 在所有的 training view 中都可见
        """
        if 'Ks' not in batch:
            __import__('ipdb').set_trace()
            return raw

        # reshape the sample points
        sh = pts.shape                      # (1, chunk, N_samples, 3), 1 <- batch_size
        pts = pts.view(sh[0], -1, sh[3])    # (1, chunk*N_samples, 3), 1 <- batch_size

        insides = []
        # batch['Ks'].size(1) == number of training views
        for nv in range(batch['Ks'].size(1)):
            # project pts to image space
            R = batch['RT'][:, nv, :3, :3]          # (1, 3, 3), 1 <- batch_size
            T = batch['RT'][:, nv, :3, 3]           # (1, 3), 1 <- batch_size
            pts_ = torch.matmul(pts, R.transpose(2, 1)) + T[:, None]        # (1, chunk*N_samples, 3)
            pts_ = torch.matmul(pts_, batch['Ks'][:, nv].transpose(2, 1))   # (1, chunk*N_samples, 3)
            # remove the influence of homogeneous coordinate
            pts2d = pts_[..., :2] / pts_[..., 2:]   # (1, chunk*N_samples, 3)

            # ensure that pts2d is inside the image
            pts2d = pts2d.round().long()
            H, W = batch['H'].item(), batch['W'].item()
            pts2d[..., 0] = torch.clamp(pts2d[..., 0], 0, W - 1)    # (1, chunk*N_samples, 3), 所有坐标点都 clamp 到 image 范围内
            pts2d[..., 1] = torch.clamp(pts2d[..., 1], 0, H - 1)    # (1, chunk*N_samples, 3), 所有坐标点都 clamp 到 image 范围内

            # remove the points outside the mask
            pts2d = pts2d[0]                        # (chunk*N_samples, 3)
            msk = batch['msks'][0, nv]              # (H, W)
            inside = msk[pts2d[:, 1], pts2d[:, 0]][None].bool() # (1, chunk*N_samples)
            insides.append(inside)

        inside = insides[0]                         # (1, chunk*N_samples)
        for i in range(1, len(insides)):
            inside = inside * insides[i]

        return inside

    def get_density_color(self, wpts, viewdir, z_vals, inside, raw_decoder):
        """ Prepare Input Sampled-Points and View Directions for NeRF
        Arguments:
            wpts        - (1, chunk, N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            viewdir     - (1, chunk, 3), normalized rays direction for each ray in this batch
            z_vals      - (1, chunk, N_samples), N_sample points' z value along the their corresponding ray direction
            inside      - (1, chunk*N_samples), if 1 说明当前 render view 的这个 sample point 在所有的 training view 中都可见
            raw_decoder - a lambda function that call the true Network.calculate_density_color()
        Returns:
            ret - {'raw' - (1, chunk*N_samples, 4), rgb color + computed alpha}
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]                 # (1, chunk, N_samples)
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)          # (1*chunk*N_samples, 3)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)    # (1*chunk*N_samples, 3)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)    # (1*chunk*N_samples,)

        # if there is no inside, return directly
        full_raw = torch.zeros([n_batch * n_pixel * n_sample, 4]).to(wpts)  # (1*chunk*N_samples, 4)
        if inside.sum() == 0:
            ret = {'raw': full_raw}
            return ret

        inside = inside.view(n_batch * n_pixel * n_sample)  # (1*chunk*N_samples,)
        wpts = wpts[inside]             # (n', 3)
        viewdir = viewdir[inside]       # (n', 3)
        dists = dists[inside]           # (n')
        ret = raw_decoder(wpts, viewdir, dists)

        full_raw[inside] = ret['raw']
        ret = {'raw': full_raw}

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch):
        """ Given ray_o, ray_d, near, far, feature_volume, Go Through NeRF to Get Density and Color
        Arguments:
            ray_o - (1, chunk, 3), rays origin in the world coordinates
            ray_d - (1, chunk, 3), rays direction in world coordinates
            near  - (1, chunk), near distance of each ray in world coordinates
            far   - (1, chunk), far distance of each ray in world coordinates
            batch - dict(), original input returned by dataloader
        Returns:
            rgb_map   - (1*chunk, 3), RGB color of rays in this batch
            acc_map   - (1*chunk,), sum of weights along each ray
            depth_map - (1*chunk,), depth of rays in this batch
        """
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)   # (1, chunk, N_samples, 3), (1, chunk, N_samples)
        # find those sample points who are inside(visible) all the training views
        # if 1 说明当前 render view 的这个 sample point 在所有的 training view 中都可见
        inside = self.prepare_inside_pts(wpts, batch)                       # (1, chunk*N_samples)

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d                                                     # (1, chunk, 3)
        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(wpts_val, viewdir_val, dists_val, batch)
        # compute the color and density, ret - {'raw' - (1, chunk*N_samples, 4), rgb color + computed alpha}
        ret = self.get_density_color(wpts, viewdir, z_vals, inside, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret = {
            'rgb_map': rgb_map.detach().cpu(),      # (1, chunk, 3)
            'acc_map': acc_map.detach().cpu(),      # (1, chunk)
            'depth_map': depth_map.detach().cpu()   # (1, chunk)
        }

        return ret

    def render(self, batch):
        # fetch needed data from current batch
        ray_o = batch['ray_o']      # (1, nrays, 3), rays origin in world coordinates, nrays 是当前 view 和 smpl 相交的光线条数
        ray_d = batch['ray_d']      # (1, nrays, 3), rays direction in world coordinates, nrays 是当前 view 和 smpl 相交的光线条数
        near = batch['near']        # (1, nrays), near distance of each ray in world coordinates, nrays 是当前 view 和 smpl 相交的光线条数
        far = batch['far']          # (1, nrays), far distance of each ray in world coordinates, nrays 是当前 view 和 smpl 相交的光线条数
        sh = ray_o.shape
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]          # 1(1 次 1 个 view 的 rendering), nrays(当前 view 和 smpl 相交的光线条数)
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
