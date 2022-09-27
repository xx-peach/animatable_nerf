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
        annots = np.load(ann_file, allow_pickle=True).item()    # eg. 'data/zju_mocap/CoreView_313/annots.npy'
        
        # 决定对 each frame 用多少个 views 作为 train input
        self.cams = annots['cams']          # all cameras' intrinsics and distortions
        num_cams = len(self.cams['K'])      # total number of cameras set in the scene
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        # camera views for this dataset according to its type
        view = cfg.training_view if split == 'train' else test_view

        # 指定 dataset 中用来训练的 frames
        i = cfg.begin_ith_frame             # default  0 for zju_mocap 313
        i_intv = cfg.frame_interval         # default  1 for zju_mocap 313
        ni = cfg.num_train_frame            # default 60 for zju_mocap 313
        # if test_novel_pose || aninerf_animation, 指定用来测试的 novel frames(poses)
        if cfg.test_novel_pose or cfg.aninerf_animation:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame
            if self.human == 'CoreView_390':
                i = 0

        # fetch the corresponding image names(train frames' train views)
        # (ni * self.num_cams, ), 就是把 training 用到的 image 路径给取出来了
        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        # fetch the corresponding camera index(train frames' train views)
        # (ni * self.num_cams, ), 就是 cfg.training_view 复制了 cfg.num_train_frame 遍
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        # number of views we use for each frame
        self.num_cams = len(view)

        # load 24 template T-pose joints' location and kinematic tree from pre-processed files
        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)                             # (24, 3), T-pose in smpl coordinates
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))  # (24,), relative relationship
        # load original blend weight of each vertex relative to each joint(vertex-wise的，和pose没有关系)
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)                           # (6890, 24), vertex[0:6890] 关于 joint[0:24] 的 weight(与 pose 无关)
        # big_poses: 在 canonical 让几何在空间分布更均匀，不然两腿之间容易学不好
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
        big_poses[8] = np.deg2rad(-angle)       # [-\pi/6, -\pi/6, -\pi/6]
        big_poses = big_poses.reshape(-1, 3)    # (24, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)        # A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of big_pose
        return big_A

    def get_mask(self, index):
        """ Read the Mask and Erode its Edge if Specified """
        # get the valid path for the mask
        msk_path = os.path.join(self.data_root, 'mask_cihp', self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask', self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace('images', 'mask'))[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace('images', 'mask'))[:-4] + '.jpg'
        # read the mask from disk into memory, of shape (H, W)
        msk_cihp = imageio.imread(msk_path)
        # change from (H, W, 3) to (H, W) since we only 2-channel mask
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        # cast the dtype into type of np.uint8
        if 'deepcap' in self.data_root or 'nerfcap' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()
        # erode the human mask's edge if cfg.erode_edge is set
        if not cfg.eval and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

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
        # read 6890 SMPL vertices xyz of this frame in the world coordinate system
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

    def __getitem__(self, index):
        """ Custom __getitem__() Function used by torch.utils.data.DataLoader
        Arguments:
            index - data(image) index that this sampler is gonna use
        Retunrs:
            ret - a dict who has all the processed data
        """
        # read the index-th original image and its human mask
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.    # (H, W, 3) in range [0, 1]
        msk, orig_msk = self.get_mask(index)                        # both of (H, W) and type uint8

        # resize the original mask and 'deepcap' mask if have
        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)              # (W, H) <- (H, W)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)    # (W, H) <- (H, W)

        # fetch the camera intrinsic and distortion, and undistort the image and mask
        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        #? why to divide the translation matrix by 1000?
        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)         # (H*ratio, W*ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)                 # (H*ratio, W*ratio)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)              # (H*ratio, W*ratio)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)    # (H*ratio, W*ratio)
        # fill the background with 0 if cfg.mask_bkgd is specified
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        # reduce cx and cy in camera intrinsic accordingly
        K[:2] = K[:2] * cfg.ratio

        # fetch the frame index from the corresponding image path
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        elif 'deepcap' in self.data_root or 'nerfcap' in self.data_root:
            i = int(os.path.basename(img_path).split('_')[-1][:-4])
            frame_index = i
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

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

        # read world smpl vertices of this frame and translate it to smpl coordinate
        # compute A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of this frame's pose, and read Rh, Th
        wpts, pvertices, A, Rh, Th, poses, nearest_frame_index = self.prepare_input(i)

        # compute current smpl vertice's bounding point, in world coordinate and smpl coordinate
        pbounds = if_nerf_dutils.get_bounds(pvertices)          # (2, 3), bounding points in pose coordinate
        wbounds = if_nerf_dutils.get_bounds(wpts)               # (2, 3), bounding points in world coordinate

        # sample nrays rays from current image if train, or all rays intersects with 3d bounding box if test
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)

        if cfg.erode_edge:
            orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]

        # nerf
        ret = {
            'rgb': rgb,                 # (nrays, 3), rgb color of each corresponding ray
            'occupancy': occupancy,     # (nrays,), whether the chosen pixel >0 or not in the original mask
            'ray_o': ray_o,             # (nrays, 3), rays origin in world coordinates
            'ray_d': ray_d,             # (nrays, 3), rays direction in world coordinates
            'near': near,               # (nrays,), near distance of each ray
            'far': far,                 # (nrays,), far distance of each ray
            'mask_at_box': mask_at_box  #!(nrays,), 对于 train 来说没用，是给 test 最后填 pixel 用的s
        }

        # blend weight
        meta = {
            'A': A,                     # (24, 4, 4), A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of current frame
            'big_A': self.big_A,        # (24, 4, 4), A(J_rel, pose) @ A(J_tpose, tpose)^{-1} of a pre-defined `big_pose`
            'poses': poses,             # (72,), input human pose of current frame after ravel() operation
            'weights': self.weights,    # (6890, 24), original blend weight of each vertex relative to each joint
            'tvertices': tvertices,     # (6890, 3), 6890 vertices of T-pose/big-pose, in world coordinate
            'pvertices': pvertices,     # (6890, 3), 6890 smpl vertices of current frame, in smpl coordinate
            'pbounds': pbounds,         # (2, 3), current vertice's bounding points in smpl coordinate
            'wbounds': wbounds,         # (2, 3), current vertice's bounding points in world coordinate
            'tbounds': tbounds          # (2, 3), T-pose/big-pose vertice's bounding points in world coordinate
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # (3, 3) R and (3,) Th of current smpl, smpl2world
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams       # 就是 frame index, 因为原本 index \in [0, num_frame*num_view-1]
        bw_latent_index = index // self.num_cams    # 就是 frame index, 因为原本 index \in [0, num_frame*num_view-1]
        if cfg.test_novel_pose or cfg.aninerf_animation:
            if 'h36m' in self.data_root:
                latent_index = 0
            else:
                latent_index = cfg.num_train_frame - 1
            latent_index = nearest_frame_index
        meta = {
            'latent_index': latent_index,           # for training step 1, global frame index
            'bw_latent_index': bw_latent_index,     # for training step 2 and test, global frame index
            'frame_index': frame_index,             # for evaluation, global truely frame index
            'cam_ind': cam_ind                      # int, view index \in [0, self.num_cams-1]
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
