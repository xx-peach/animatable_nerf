import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = cfg.result_dir     # 'data/result'
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def visualize_image(self, output, batch):
        # compute the mse between the predicted and ground truth rgb
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()          # (n, 3), n 是指 H*W 个 pixels 里面发出的光线和 smpl 相交的个数
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()                 # (n, 3), n 是指 H*W 个 pixels 里面发出的光线和 smpl 相交的个数
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt)**2)))

        # fetch `mask_at_box` from original rays sampling
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()    # (nrays,) == (H * W,)
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)                         # (H, W)

        # get predicted images using `mask_at_box`
        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        # get ground truth images using `mask_at_box`
        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        # create shub-directory for predicted images and ground truth images
        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        # save the predicted images and ground truth images using cv2.imwrite()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index, view_index), (img_pred[..., [2, 1, 0]] * 255)
        )
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index, view_index), (img_gt[..., [2, 1, 0]] * 255)
        )

        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(img_pred)
        # ax2.imshow(img_gt)
        # plt.show()

    def visualize_normal(self, output, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)
        surf_mask = mask_at_box.copy()
        surf_mask[mask_at_box] = output['surf_mask'][0].detach().cpu().numpy()

        normal_map = np.zeros((H, W, 3))
        normal_map[surf_mask] = output['surf_normal'][
            output['surf_mask']].detach().cpu().numpy()

        normal_map[..., 1:] = normal_map[..., 1:] * -1
        norm = np.linalg.norm(normal_map, axis=2)
        norm[norm < 1e-8] = 1e-8
        normal_map = normal_map / norm[..., None]
        normal_map = (normal_map + 1) / 2

        plt.imshow(normal_map)
        plt.show()

    def visualize_acc(self, output, batch):
        acc_pred = output['acc_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        acc = np.zeros((H, W))
        acc[mask_at_box] = acc_pred

        plt.imshow(acc)
        plt.show()

        # acc_path = os.path.join(cfg.result_dir, 'acc')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # acc_path = os.path.join(acc_path, '{:04d}_{:02d}.jpg'.format(i, cam_ind))
        # os.system('mkdir -p {}'.format(os.path.dirname(acc_path)))
        # plt.savefig(acc_path)

    def visualize_depth(self, output, batch):
        depth_pred = output['depth_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        depth = np.zeros((H, W))
        depth[mask_at_box] = depth_pred

        plt.imshow(depth)
        plt.show()

        # depth_path = os.path.join(cfg.result_dir, 'depth')
        # i = batch['i'].item()
        # cam_ind = batch['cam_ind'].item()
        # depth_path = os.path.join(depth_path, '{:04d}_{:02d}.jpg'.format(i, cam_ind))
        # os.system('mkdir -p {}'.format(os.path.dirname(depth_path)))
        # plt.savefig(depth_path)

    def visualize(self, output, batch):
        self.visualize_image(output, batch)
