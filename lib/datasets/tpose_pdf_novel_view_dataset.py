import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import render_utils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        """ Dataset __init__() Function
        Args:
            data_root - str, the root data directory, eg. 'data/zju_mocap/CoreView_313'
            human     - str, which specific human's data we're gonna use, eg. 'CoreView_313'
            ann_file  - str, the path for 'params.npy' which stores each frame's smpl parameters, including
                        beta, pose and tran, eg. 'data/zju_mocap/CoreView_313/annots.npy'
            split     - str, ['train', 'test'], indicating which kind of dataset it is
        """
        super(Dataset, self).__init__()

        # fetch basic configuration for this dataset
        self.data_root = data_root          # eg. 'data/zju_mocap/CoreView_313'
        self.human = human                  # eg. 'CoreView_313'
        self.split = split                  # ['train', 'test']
        annots = np.load(ann_file, allow_pickle=True).item()        # eg. 'data/zju_mocap/CoreView_313/annots.npy'
        
        # determine visualization views
        self.cams = annots['cams']          # all cameras' intrinsics and distortions
        test_view = [3]                     #? pre-defined visualization view [3], 在这个 novel view vis 里没啥用
        view = cfg.training_view if split == 'train' else test_view # split == 'test'
        self.num_cams = len(view)           #? total number of cameras set in the scene, 没啥用
        
        # fetch all cameras' intrinsic and extrinsic from parameter file 'ann_file'
        K, RT = render_utils.load_cam(ann_file)     # lists of (3, 3) camera intrinsic and (4, 4) extrinsic matrices, world2camera
        # generate spiral novel views for visualization
        render_w2c = render_utils.gen_path(RT)      # list of cfg.render_views (3, 3) world2camera transformation matrix

        # determine the novel visualization frame, from frame[0] to frame[num_train_frame * frame_interval]
        # (num_train_frame * frame_interval, training_view), 把 training(其实是 test) 要用的 image path 给取出来
        i = cfg.begin_ith_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][:cfg.num_train_frame * cfg.frame_interval]
        ])

        self.K = K[0]                       # camera[0] 的 intrinsic matrix
        self.render_w2c = render_w2c        # list of cfg.render_views (3, 3) world2camera transformation matrix
        img_root = 'data/render/{}'.format(cfg.exp_name)
        # base_utils.write_K_pose_inf(self.K, self.render_w2c, img_root)

        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)         # camera2image
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)        # world2camera
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(np.float32)

        # load 24 template T-pose joints' location and kinematic tree from pre-processed files
        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)                             # (24, 3), T-pose in smpl coordinates
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))  # (24,), relative relationship
        # load original blend weight of each vertex relative to each joint in T-pose
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)                           # (6890, 24), vertex[0:6890] 关于 joint[0:24] 的 weight(与 pose 无关)
        # big_poses: 在 canonical 让几何在空间分布更均匀，不然两腿之间的容易学不好
        self.big_A = self.load_bigpose()                                    # (24, 4, 4), A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of bigpose
        
        # test novel pose 的时候要找与 novel pose 最接近的 training pose(其 joints 记录在了 'training_joints.npy' 中)
        if cfg.test_novel_pose or cfg.aninerf_animation:
            training_joints_path = os.path.join(self.lbs_root, 'training_joints.npy')
            if os.path.exists(training_joints_path):
                self.training_joints = np.load(training_joints_path)
        
        self.nrays = cfg.N_rand

    def load_bigpose(self):
        """ Compute A(J_rel, pose) @ A(J_tpose, tpose)^{-1} for a Defined `big_poses` """
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)        # [ \pi/6,  \pi/6,  \pi/6]
        big_poses[8] = np.deg2rad(-angle)       # [ \pi/6,  \pi/6,  \pi/6]
        big_poses = big_poses.reshape(-1, 3)    # (24, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)        # A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of big_pose
        return big_A

    def prepare_input(self, i):
        """ Transform 6890 Vertices of Current Frame's SMPL from World to SMPL
        Arguments:
            i            - data(image) index that this sampler is gonna use, 不是 __getitem__ 收到的 index, 而是从 image path 中取的
            cfg.vertices - str(), 'new_vertices' directory
        Returns:
            wxyz  - (6890, 3) of float32, 6890 smpl vertices of this frame, in world coordinate
            pxyz  - (6890, 3) of float32, 6890 smpl vertices of this frame, in smpl coordinate
            A     - (24, 4, 4) of float32, G(poses, j_rel) * G(zero_pose, j)^{-1} of this frame's pose
            Rh    - (3,) of float32, angle-axis form before Rodrigues, smpl2world
            Th    - (3,) of float32, translation vector, smpl2world
            poses - (72,) of float32, namely input human pose of current frame after ravel() operation
            nearest_frame_index - int, 这是 test 时找一个和 novel frame pose 最接近的，在 NeRF 计算时用那个 frame 的 latent code
        """
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1
        # read current frame's xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices, '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices, '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params, '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, cfg.params, '{:06d}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)                # (3,), Rh to represents the global oritentation
        Th = params['Th'].astype(np.float32)                # (3,), Th to represents the global translation

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)         # (3, 3), transfer from angle-axis form to 3x3 rotation matrix
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)      # (6890, 3), vertices in smpl coordinate

        # calculate the skeleton transformation of this frame's human pose
        poses = params['poses'].reshape(-1, 3)              # (24, 3)
        joints = self.joints                                # (24, 3), T-pose in smpl coordinate
        parents = self.parents                              # (24,), pre-defined kinematic tree
        # A: A(J_rel, pose) @ A(J_tpose, tpose)^{-1}, of shape (24, 4, 4)
        A, canonical_joints = if_nerf_dutils.get_rigid_transformation(poses, joints, parents, return_joints=True)

        # transform joint location under the input pose in smpl coordinate to world coordinate
        posed_joints = np.dot(canonical_joints, R.T) + Th   # (24, 3), 为下面 test 时找 nearest training frame 用

        ##############################################################################
        # find the nearest training frame, 这是 test 时 NeRF 要找一个最相近的 \phi_i 用的 #
        ##############################################################################
        if (cfg.test_novel_pose or cfg.aninerf_animation) and hasattr(self, "training_joints"):
            nearest_frame_index = np.linalg.norm(self.training_joints -
                                                 posed_joints[None],
                                                 axis=2).mean(axis=1).argmin()
        # if train, we don't need this nearest_frame_index at all
        else:
            nearest_frame_index = 0

        # ravel the poses and cast its type into np.float32
        poses = poses.ravel().astype(np.float32)            # (72,)

        return wxyz, pxyz, A, Rh, Th, poses, nearest_frame_index

    def get_mask(self, i):
        """ Read the Corresponding Masks of Frame[i]'s All Test Views """
        ims = self.ims[i]       # (num_train_frame * frame_interval, training_view), array of all test views of frame[i]
        msks = []               # empty mask list reserved for corresponding masks

        for nv in range(len(ims)):
            # nv-th view of this frame
            im = ims[nv]
            # get the valid path for the mask
            msk_path = os.path.join(self.data_root, 'mask_cihp', im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, 'mask', im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, im.replace('images', 'mask'))[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, im.replace('images', 'mask'))[:-4] + '.jpg'
            # read the mask from disk into memory, of shape (H, W)
            msk_cihp = imageio.imread(msk_path)
            
            # change from (H, W, 3) to (H, W) since we only 2-channel mask
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            # cast the dtype into type of np.uint8
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
            msk = msk_cihp.astype(np.uint8)

            # undistort the corresponding mask
            K = self.Ks[nv].copy()
            K[:2] = K[:2] / cfg.ratio
            msk = cv2.undistort(msk, K, self.Ds[nv])
            # dilate the corresponding mask
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        """ Custom __getitem__() Function used by torch.utils.data.DataLoader
        Arguments:
            index - data(image) index that this sampler is gonna use
        Retunrs:
            ret - a dict who has all the processed data
        """
        view_index = index                                      # [0, cfg.render_views], 当前 batch 要 render 的第 index 个 novel view
        latent_index = cfg.begin_ith_frame                      # [0, cfg.num_train_frame], network 的 latent code, 所以不用考虑 views 的关系
        frame_index = cfg.begin_ith_frame * cfg.frame_interval  # [0, cfg.num_train_frame*cfg.frame_interval], 读取当前 test frame 的 vertices 用的，所以要考虑 interval 得到真正的 frame index

        # https://blog.csdn.net/weixin_38705903/article/details/79231551
        if cfg.get('use_bigpose', False):
            # if we specify use_bigpose == True in the config
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            # if we do not specify use_bigpose or specify use_bigpose == False in the config
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        # read vertices of the T-pose/big-pose template, and compute its bounds in world coordinate
        tvertices = np.load(vertices_path).astype(np.float32)   # (6890, 3)
        tbounds = if_nerf_dutils.get_bounds(tvertices)          # (2, 3)

        # read world smpl vertices of frame[frame_index] and translate it to smpl coordinate
        # compute A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of frame[frame_index]'s pose, and read its Rh, Th
        wpts, pvertices, A, Rh, Th, poses, nearest_frame_index = self.prepare_input(frame_index)
        # compute frame[frame_index]'s smpl bounding point, in world coordinate and smpl coordinate
        pbounds = if_nerf_dutils.get_bounds(pvertices)          # (2, 3), bounding points in pose coordinate
        wbounds = if_nerf_dutils.get_bounds(wpts)               # (2, 3), bounding points in world coordinate

        # read frame[frame_index] 的 human mask
        msks = self.get_mask(frame_index)                       # (v, H, W)

        #? read frame[0] 的 view[0] 的 original image --> 只是为了得到 image 的 H, W
        img_path = os.path.join(self.data_root, self.ims[0][0])
        img = imageio.imread(img_path)                          # (H, W, 3)
        # reduce the original image and mask's resolution by ratio
        H, W = img.shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        msks = [
            cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST) for msk in msks
        ]
        msks = np.array(msks)                                   # (v, H, W)

        # sample rays that within bounds from frame[frame_index] under index-th novel view to render
        K = self.K                          # camera[0] 的 intrinsic matrix
        RT = self.render_w2c[index]         # 第 index 个 novel view 的 world2camera matrix
        R, T = RT[:3, :3], RT[:3, 3:]       # seperate (4, 4) render_w2c to (3, 3) rotation and (3, 1) translation matrix
        # call `get_rays_within_bounds()` to generate each pixel's ray, all of shape (n, ) except mask_at_box of (H, W)
        # 该函数和 `sample_rays_h36m()` 里面的 test branch 部分一样, 就是先生成 all H*W 条 rays, 然后在计算 near, far 的时候计算相交来筛选
        ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds(H, W, K, R, T, wbounds)
        # ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(RT, K, wbounds)

        # nerf
        ret = {
            'ray_o': ray_o,             # (n, 3), rays origin in world coordinates, n < H*W, 是和 smpl 相交的光线条数
            'ray_d': ray_d,             # (n, 3), rays direction in world coordinates, n < H*W, 是和 smpl 相交的光线条数
            'near': near,               # (n,), near distance of each ray, n < H*W, 是和 smpl 相交的光线条数
            'far': far,                 # (n,), far distance of each ray, n < H*W, 是和 smpl 相交的光线条数
            'mask_at_box': mask_at_box  # (H, W), mask matrix, mask_at_box[i, j] == True if the rays of (i, j) intersects with the smpl
        }

        # blend weight
        meta = {
            'A': A,                     # (24, 4, 4), A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of frame[frame_index]
            'big_A': self.big_A,        # (24, 4, 4), A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of a pre-defined `big_pose`
            'poses': poses,             # (72,), input human pose of frame[frame_index] after ravel() operation
            'weights': self.weights,    # (6890, 24), original blend weight of each vertex relative to each joint
            'tvertices': tvertices,     # (6890, 3), 6890 vertices of T-pose/big-pose, in world coordinate
            'pvertices': pvertices,     # (6890, 3), 6890 smpl vertices of frame[frame_index], in smpl coordinate
            'pbounds': pbounds,         # (2, 3), frame[frame_index]'s vertice bounding points in smpl coordinate
            'wbounds': wbounds,         # (2, 3), frame[frame_index]'s vertice bounding points in world coordinate
            'tbounds': tbounds          # (2, 3), T-pose/big-pose vertice's bounding points in world coordinate
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'R': R,                         # (3, 3) global rotation matrix of frame[frame_index], smpl2world
            'Th': Th,                       # (3,) global translation matrix of frame[frame_index], smpl2world
            'latent_index': latent_index,   # [0, cfg.num_train_frame], network 的 latent code, 所以不用考虑 views 的关系
            'frame_index': frame_index,     # [0, cfg.num_train_frame*cfg.frame_interval], 读取当前 test frame 的 vertices 用的，所以要考虑 interval 得到真正的 frame index
            'view_index': view_index        # [0, cfg.render_views], 当前 batch 要 render 的第 index 个 novel view
        }
        ret.update(meta)

        meta = {'msks': msks, 'Ks': self.Ks, 'RT': self.RT, 'H': H, 'W': W}
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.render_w2c)     # cfg.render_views
