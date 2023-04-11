import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from .loading_nusc_occ import custom_rotate_3d
import pdb

@PIPELINES.register_module()
class LoadSemKittiAnnotation():
    def __init__(self, bda_aug_conf, is_train=True, 
                 point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""

        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf['flip_dz_ratio']
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def forward_test(self, results):
        bda_rot = torch.eye(4).float()
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
        
        return results

    def __call__(self, results):
        if results['gt_occ'] is None:
            return self.forward_test(results)
        
        if type(results['gt_occ']) is list:
            gt_occ = [torch.tensor(x) for x in results['gt_occ']]
        else:
            gt_occ = torch.tensor(results['gt_occ'])
        
        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_bda_augmentation()
            gt_occ, bda_rot = voxel_transform(gt_occ, rotate_bda, scale_bda, 
                        flip_dx, flip_dy, flip_dz, self.transform_center)
        else:
            bda_rot = torch.eye(4).float()
        
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
        results['gt_occ'] = gt_occ.long()
        
        return results

def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz, transform_center=None):
    # for semantic_kitti, the transform origin is not zero, but the center of the point cloud range
    assert transform_center is not None
    trans_norm = torch.eye(4)
    trans_norm[:3, -1] = - transform_center
    trans_denorm = torch.eye(4)
    trans_denorm[:3, -1] = transform_center
    
    # bird-eye-view rotation
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    # I @ flip_x @ flip_y
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dz:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, 1]])
    
    # denorm @ flip_x @ flip_y @ flip_z @ rotation @ normalize
    bda_mat = trans_denorm @ flip_mat @ rot_mat @ trans_norm
    
    voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    if not np.isclose(rotate_degree, 0):
        voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)
    
    if flip_dz:
        voxel_labels = voxel_labels[:, :, ::-1]
    
    if flip_dy:
        voxel_labels = voxel_labels[:, ::-1]
    
    if flip_dx:
        voxel_labels = voxel_labels[::-1]
    
    voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat