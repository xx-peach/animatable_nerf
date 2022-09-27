import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg
import trimesh


# def get_training_joints(num_train_frame, frame_interval, data_root):
#     def get_joints(frame_index):
#         inds = os.listdir(param_root)
#         inds = sorted([int(ind[:-4]) for ind in inds])
#         frame_index = inds[frame_index]

#         # transform smpl from the world coordinate to the smpl coordinate
#         params_path = os.path.join(param_root, '{}.npy'.format(frame_index))
#         params = np.load(params_path, allow_pickle=True).item()
#         Rh = params['Rh'].astype(np.float32)
#         Th = params['Th'].astype(np.float32)

#         # prepare sp input of param pose
#         R = cv2.Rodrigues(Rh)[0].astype(np.float32)

#         # calculate the skeleton transformation
#         poses = params['poses'].reshape(-1, 3)
#         A, canonical_joints = get_rigid_transformation(
#             poses, joints, parents, return_joints=True)

#         posed_joints = np.dot(canonical_joints, R.T) + Th

#         return posed_joints

#     lbs_root = os.path.join(data_root, 'lbs')
#     joints = np.load(os.path.join(lbs_root, 'joints.npy'))
#     joints = joints.astype(np.float32)
#     parents = np.load(os.path.join(lbs_root, 'parents.npy'))
#     param_root = os.path.join(data_root, 'new_params')

#     training_joints = []
#     for i in range(0, num_train_frame * frame_interval, frame_interval):
#         posed_joints = get_joints(i)
#         training_joints.append(posed_joints)
#     training_joints = np.stack(training_joints)

#     np.save(os.path.join(lbs_root, 'training_joints.npy'), training_joints)
#     return training_joints


def get_rays_within_bounds_test(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)
    far = far.astype(np.float32)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)

    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T):
    """ Function for Ray Generation, Matrix Multiplicaion
    Arguments:
        H, W - int, the height and width of the input image
        K    - (3, 3) of float32 the camera intrinsic matrix
        R    - (3, 3) of float32, rotation matrix from world to camera
        T    - (3,) of float32, translation matrix from world to camera
    Returns:
        rays_o - (H, W, 3), duplication of (3, ) camera origin in world coordinate
        rays_d - (H, W, 3), camera directions in world coordinate, same as NeRF
    """
    # calculate the camera origin, RX + T = 0 -> X = R.inv @ -T
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels using `np.meshgrid()`
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)     # (H, W, 3)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)      # (H, W, 3), same as K.inv @ xy1.T
    pixel_world = np.dot(pixel_camera - T.ravel(), R)   # (H, W, 3)
    #! calculate the ray direction, 下面减 rays_o 是因为上面是计算了 pixel_world 而不是只是乘了 R
    rays_d = pixel_world - rays_o[None, None]           # (H, W, 3)
    # normalize the rays' direction directly here
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)      # (H, W, 3)
    return rays_o, rays_d


def get_bound_corners(bounds):
    """ Generate 3d Bounding Box's 8 Vertices Coordinate(World)
    Arguments:
        bounds - (2, 3) of float32, human vertices bound in world coordinates
    Returns:
        corners_3d - (8, 3) of float32, 3d bounding box coordinates of 6890 smpl vertices
    """
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    """ Generate 2d Image Mask Projected from 3d Vertices Bouning Box
    Arguments:
        bounds - (2, 3) of float32, human vertices bound in world coordinates
        K      - (3, 3) of float32, camera's intrinsic matrix
        pose   - (3, 4) of float32, project matrix from world to camera
        H, W   - int, the height and width of the input image
    Returns:
        mask - (H, W) of 0/1, 2d human mask projected from 3d smpl vertices bounding box
    """
    corners_3d = get_bound_corners(bounds)                  # (8, 3), 3d bounding box
    corners_2d = base_utils.project(corners_3d, K, pose)    # (8, 3), project 3d bounding box to image
    corners_2d = np.round(corners_2d).astype(int)           # (8, 3), rounded to int
    mask = np.zeros((H, W), dtype=np.uint8)
    # use `cv2.fillPoly()` to draw the mask of 2d bounding box projected from 3d world bounding box
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)    # verticel left plane
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)    # verticle right plane
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)    # verticle back plane
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)    # verticle front plane
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)    # bottom plane
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)    # top plane
    return mask


# def get_near_far(bounds, ray_o, ray_d):
#     """calculate intersections with 3d bounding box"""
#     norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
#     viewdir = ray_d / norm_d
#     viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
#     viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
#     tmin = (bounds[:1] - ray_o[:1]) / viewdir
#     tmax = (bounds[1:2] - ray_o[:1]) / viewdir
#     t1 = np.minimum(tmin, tmax)
#     t2 = np.maximum(tmin, tmax)
#     near = np.max(t1, axis=-1)
#     far = np.min(t2, axis=-1)
#     mask_at_box = near < far
#     near = near[mask_at_box] / norm_d[mask_at_box, 0]
#     far = far[mask_at_box] / norm_d[mask_at_box, 0]
#     return near, far, mask_at_box


def get_near_far(bounds, ray_o, ray_d):
    """ Calculate Intersections with 3D Bounding Box Using 3D Slabs Principle
        https://blog.csdn.net/weixin_40301728/article/details/114239266
    Args:
        bounds - (2, 3) of float32, human vertices bound in world coordinates
        ray_o  - (nbody+nrand, 3) / (H*W, 3), duplication of (3, ) camera origin in world coordinate
        ray_d  - (nbody+nrand, 3) / (H*W, 3), rays directions of each pre-sampled points(这里还会做筛选)
    Returns:
        near        - (n,) of float32, near distance for those rays intersecting with the 3d bounding box
        far         - (n,) of float32, far distance for those rays intersecting with the 3d bounding box
        mask_at_box - (nbody+nrand,) of int32, 1 if that ray intersects with the 3d bounding box
    """
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]                               # (n, 2, 3)
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)               # (n, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]  # (n, 6, 3)
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))                  # (n, 6)
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2                                # (n,)
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(-1, 2, 3)    # (n', 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]                                              # (n', 3)
    ray_d = ray_d[mask_at_box]                                              # (n', 3)
    norm_ray = np.linalg.norm(ray_d, axis=1)                                # (n',)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray       # (n',)
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray       # (n',)
    near = np.minimum(d0, d1)                                               # (n',)
    far = np.maximum(d0, d1)                                                # (n',)

    return near, far, mask_at_box


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
    """ Sample Rays from Current Image(['train', 'test'])
    Arguments:
        img    - (H, W, 3) of float32, original image of this batch
        msk    - (H, W, 3) of float32, masked image of the batch
        K      - (3, 3) of float32, camera's intrinsic matrix
        R      - (3, 3) of float32, rotation matrix from world to camera
        T      - (3,) of float32, translation matrix from world to camera
        bounds - (2, 3) of float32, human vertices bound in world coordinates
        nrays  - int, namely choose how many rays for this batch
        split  - ['train', 'test']
    Returns:
        rgb         - (nrays, 3), rgb color of each corresponding ray
        ray_o       - (nrays, 3), rays origin in world coordinates
        ray_d       - (nrays, 3), rays direction in world coordinates
        near        - (nrays,), near distance of each ray
        far         - (nrays,), far distance of each ray
        coord       - (nrays, 2), correspoding image indices (u, v) of n rays
        mask_at_box - #? (nrays,), 都是 1 的一个 array...
    """
    # get H*W rays' origin and direction
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)                  # (H, W, 3)

    # generate 2d human mask which is projected from 3d smpl vertices bounding box
    pose = np.concatenate([R, T], axis=1)                   # (3, 4)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)   # (H, W) of 0/1

    if cfg.mask_bkgd: img[bound_mask != 1] = 0
    # mask the original input human mask further
    msk = msk * bound_mask
    bound_mask[msk == 100] = 0

    # sample nrays(batch_size) rays from H*W rays if it is train loader
    if split == 'train':
        nsampled_rays = 0                           # number of rays we've sampled
        face_sample_ratio = cfg.face_sample_ratio   # face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio   # body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            # specify number of rays sampled from human body, face and bound mask just generated
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)   # num of rays sampled from human body
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)   # num of rays sampled from human face
            n_rand = (nrays - nsampled_rays) - n_body - n_face          # num of rays remaining, sampled from bound mask

            # sample rays on body, msk == 1's places are all human body
            coord_body = np.argwhere(msk == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body), n_body)]
            # sample rays on face, msk == 13's place human face, 没用到其实
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask from bound_mask that just generated
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            # concatenate all the sampled rays indices btw [[0, 0], ... [H-1, W-1]]
            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            # fetch ray_o, ray_d, and corresponding rgb using sampled indices
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]    # (nbody+nrand, 3)
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]    # (nbody+nrand, 3)
            rgb_   =   img[coord[:, 0], coord[:, 1]]    # (nbody+nrand, 3)

            # generate near, far distance for each sampled rays, and further filter rays
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)     # (n,)

            ray_o_list.append(ray_o_[mask_at_box])              # (n, 3), rays origin in world coordinates
            ray_d_list.append(ray_d_[mask_at_box])              # (n, 3), rays direction in world coordinates
            rgb_list.append(rgb_[mask_at_box])                  # (n, 3), rgb color of each corresponding ray
            near_list.append(near_)                             # (n,), near distance of each ray
            far_list.append(far_)                               # (n,), far distance of each ray
            coord_list.append(coord[mask_at_box])               # (n, 2), correspoding image indices (u, v) of n rays
            mask_at_box_list.append(mask_at_box[mask_at_box])   #* 这样就是全 1 了, 不过没事, mask 是给 visualize 用的
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    
    # generate H*W rays and filter those who intersects with the 3d bounding box if it's test loader
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_rays_within_bounds(H, W, K, R, T, bounds):
    """ Used by Dataset When Testing Novel Views
    Arguments:
        H, W   - int, the height and width of the input image
        K      - (3, 3) of float32, camera's intrinsic matrix
        R      - (3, 3) of float32, rotation matrix from world to camera
        T      - (3,) of float32, translation matrix from world to camera
        bounds - (2, 3) of float32, human vertices bound in world coordinates
    Returns:
        ray_o - (n, 3), rays origin in world coordinates
        ray_d - (n, 3), rays direction in world coordinates
        near  - (n,), near distance of each ray
        far   - (n,), far distance of each ray
        mask_at_box - (H, W), 和 smpl 相交的 pixel 位置是 True
    """
    # get all pixels' corresponding rays
    ray_o, ray_d = get_rays(H, W, K, R, T)          # (H, W, 3)
    # compute intersection and near, far plane
    ray_o = ray_o.reshape(-1, 3).astype(np.float32) # (H*W, 3)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32) # (H*W, 3)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
    near = near.astype(np.float32)                  # (n,), 这个 n 是指 H*W 个 pixels 里面发出的光线和 smpl 相交的个数
    far = far.astype(np.float32)                    # (n,), 这个 n 是指 H*W 个 pixels 里面发出的光线和 smpl 相交的个数
    # filter rays that do not intersect
    ray_o = ray_o[mask_at_box]                      # (n,), 这个 n 是指 H*W 个 pixels 里面发出的光线和 smpl 相交的个数
    ray_d = ray_d[mask_at_box]                      # (n,), 这个 n 是指 H*W 个 pixels 里面发出的光线和 smpl 相交的个数
    # reshape the mask
    mask_at_box = mask_at_box.reshape(H, W)         # (H, W), 和 smpl 相交的 pixel 位置是 True

    return ray_o, ray_d, near, far, mask_at_box


def get_acc(coord, msk):
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def sample_world_points(ray_o, ray_d, near, far, split):
    # calculate the steps for each ray
    t_vals = np.linspace(0., 1., num=cfg.N_samples)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.astype(np.float32)
    z_vals = z_vals.astype(np.float32)

    return pts, z_vals


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents, return_joints=False):
    """ Compute G = A(J_rel, pose) @ A(J_tpose, tpose)^{-1} in a Vectorized Method
    Arguments:
        poses   - (24, 3), poses[0] is [0, 0, 0](note that it is not the global pose because we have a separate 'Rh' in
                  ZJU-MoCap dataset to represent global pose), and poses[1:] are poses relative to their parent
        joints  - (24, 3), 24 joints' location of T-pose model, in smpl coordinate, x_mean == y_mean == z_mean == 0
        parents - (24,), pre-defined kinematic tree including 24 joints relative relationship
    Returns:
        transforms   - (24, 4, 4), namely A(J_rel, pose) @ A(J_tpose, tpose)^{-1}, pose to smpl coordinate
        posed_joints - (24, 3), joint location under the input pose in smpl coordinate
    """
    # transfer from angle-axis form to 3x3 rotation matrix form
    rot_mats = batch_rodrigues(poses)                                   # (24, 3, 3)
    # obtain the relative joints locations for A(J_rel, pose) for joint[1:23]
    rel_joints = joints.copy()                                          # (24, 3)
    rel_joints[1:] -= joints[parents[1:]]                               # (24, 3)

    # create the 4x4 relative transformation matrics
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)  # (24, 3, 4)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)          # (24, 4, 4)

    # backtrack through the kinematic tree to rotate each part and get G(poses, j_rel)
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)                      # (24, 4, 4)

    # get the joint location under the input pose(没有 remove the transformation due to T-pose)
    posed_joints = transforms[:, :3, 3].copy()                          # (24, 3)

    # obtain the rigid transformation after removing T-pose and get G(poses, j_rel) * G(zero_pose, j)^{-1}
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)          # (24, 4)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)   # (24, 4, 4) * (24, 4, 1)
    # remove the transformation due to T-pose(rest pose), 看 notion SMPL 最后 5.1 里面的推导
    transforms[..., 3] = transforms[..., 3] - rel_joints                # (24, 4, 4)
    transforms = transforms.astype(np.float32)                          # (24, 4, 4)

    if return_joints:
        return transforms, posed_joints
    else:
        return transforms


def padding_bbox(bbox, img):
    padding = 10
    bbox[0] = bbox[0] - 10
    bbox[1] = bbox[1] + 10

    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.5

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    h, w = img.shape[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def crop_image_msk(img, msk, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, img)

    crop = img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    crop_msk = msk[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    # calculate the shape
    shape = crop.shape
    x = 8
    height = (crop.shape[0] | (x - 1)) + 1
    width = (crop.shape[1] | (x - 1)) + 1

    # align image
    aligned_image = np.zeros([height, width, 3])
    aligned_image[:shape[0], :shape[1]] = crop
    aligned_image = aligned_image.astype(np.float32)

    # align mask
    aligned_msk = np.zeros([height, width])
    aligned_msk[:shape[0], :shape[1]] = crop_msk
    aligned_msk = (aligned_msk == 1).astype(np.uint8)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return aligned_image, aligned_msk, K, bbox


def random_crop_image(img, msk, K, min_size=80, max_size=88):
    H, W = img.shape[:2]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    max_size = min_HW
    min_size = int(min(min_size, 0.8 * min_HW))
    H_size = np.random.randint(min_size, max_size)
    W_size = H_size
    x = 8
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    coord = np.argwhere(msk == 1)
    center_xy = coord[np.random.randint(0, len(coord))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    img = img[begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    msk = msk[begin_y:begin_y + H_size, begin_x:begin_x + W_size]

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - begin_x
    K[1, 2] = K[1, 2] - begin_y
    K = K.astype(np.float32)

    return img, msk, K


def get_bounds(xyz):
    """ Compute two Bounding Points of 6890 SMPL Vertices
    Arguments:
        xyz - (6890, 3), 6890 SMPL vertices in whatever coordinate
    Returns:
        bounds - (2, 3), two bounding points of the SMPL vertices
    """
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= cfg.box_padding
    max_xyz += cfg.box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds


def prepare_sp_input(xyz):
    # obtain the bounds for coord construction
    bounds = get_bounds(xyz)
    # construct the coordinate
    dhw = xyz[:, [2, 1, 0]]
    min_dhw = bounds[0, [2, 1, 0]]
    max_dhw = bounds[1, [2, 1, 0]]
    voxel_size = np.array(cfg.voxel_size)
    coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)
    # construct the output shape
    out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    return coord, out_sh, bounds


def crop_mask_edge(msk):
    msk = msk.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk


def adjust_hsv(img, saturation, brightness, contrast):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv[..., 1] = np.minimum(hsv[..., 1], 255)
    hsv[..., 2] = hsv[..., 2] * brightness
    hsv[..., 2] = np.minimum(hsv[..., 2], 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) * contrast
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img
