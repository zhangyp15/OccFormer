#import open3d as o3d
import numpy as np
import yaml, os
import torch
import torch.nn.functional as F
import numba as nb

from PIL import Image
from mmdet.datasets.builder import PIPELINES

import pdb

@PIPELINES.register_module()
class LoadNuscOccupancyAnnotations(object):
    def __init__(
            self,
            data_root='data/nuscenes',
            is_train=False,
            is_test_submit=False,
            grid_size=None, 
            point_cloud_range=None,
            bda_aug_conf=None,
            unoccupied_id=17,
            cls_metas='nuscenes.yaml',
        ):
        
        self.is_train = is_train
        self.is_test_submit = is_test_submit
        self.cls_metas = cls_metas
        with open(cls_metas, 'r') as stream:
            nusc_cls_metas = yaml.safe_load(stream)
            self.learning_map = nusc_cls_metas['learning_map']
        
        self.data_root = data_root
        self.bda_aug_conf = bda_aug_conf
        
        # voxel settings
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = np.array(point_cloud_range)
        # for semantickitti, the transformer center is not (0, 0, 0) and makes the transformation a bit more complex
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2
        self.unoccupied_id = unoccupied_id
        
        # create full-resolution occupancy labels
        self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size
    
    def sample_3d_augmentation(self):
        """Generate 3d augmentation values based on bda_config."""
        
        # Currently, we only use the flips along three directions. The rotation and scaling are not fully experimented.
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def __call__(self, results):
        # for test-submission of nuScenes LiDAR Segmentation 
        if self.is_test_submit:
            imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
            bda_rot = torch.eye(3).float()
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
            
            pts_filename = results['pts_filename']
            points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
            points_label = np.zeros((points.shape[0], 1)) # placeholder
            lidarseg = np.concatenate([points, points_label], axis=-1)
            results['points_occ'] = torch.from_numpy(lidarseg).float()
            
            return results
        
        ''' load lidarseg points '''
        lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        pts_filename = results['pts_filename']
        
        points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
        lidarseg = np.concatenate([points, points_label], axis=-1)
        
        ''' create multi-view projections '''
        ''' apply 3D augmentation for lidar_points (and the later generated occupancy) '''
        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_3d_augmentation()
            _, bda_rot = voxel_transform(None, rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz)
        else:
            bda_rot = torch.eye(3).float()
        
        # transform points
        points = points @ bda_rot.t().numpy()
        lidarseg[:, :3] = points
        
        ''' create voxel labels from lidarseg '''
        eps = 1e-5
        points_grid_ind = np.floor((np.clip(lidarseg[:, :3], self.point_cloud_range[:3],
                self.point_cloud_range[3:] - eps) - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int)
        
        label_voxel_pair = np.concatenate([points_grid_ind, lidarseg[:, -1:]], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((points_grid_ind[:, 0], points_grid_ind[:, 1], points_grid_ind[:, 2])), :]
        label_voxel_pair = label_voxel_pair.astype(np.int64)
        
        # 0: noise, 1-16 normal classes, 17 unoccupied (empty)
        empty_id = self.unoccupied_id
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * empty_id
        processed_label = nb_process_label(processed_label, label_voxel_pair)
        
        # convert label_0 to label_255 (ignored)
        processed_label[processed_label == 0] = 255
        # convert empty to label id 0
        processed_label[processed_label == empty_id] = 0
        
        # output: bda_mat, point_occ, and voxel_occ
        results['gt_occ'] = torch.from_numpy(processed_label).long()
        results['points_occ'] = torch.from_numpy(lidarseg).float()
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)

        return results
    
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz):
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
    bda_mat = flip_mat @ rot_mat
    bda_mat = bda_mat[:3, :3]
    
    # apply transformation to the 3D volume, which is tensor of shape [X, Y, Z]
    if voxel_labels is not None:
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
        if not np.isclose(rotate_degree, 0):
            '''
            Currently, we use a naive method for 3D rotation because we found the visualization of 
            rotate results with scipy is strange: 
                scipy.ndimage.interpolation.rotate(voxel_labels, rotate_degree, 
                        output=voxel_labels, mode='constant', order=0, 
                        cval=255, axes=(0, 1), reshape=False)
            However, we found further using BEV rotation brings no gains over 3D flips only.
            '''
            voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)
        
        if flip_dz:
            voxel_labels = voxel_labels[:, :, ::-1]
        
        if flip_dy:
            voxel_labels = voxel_labels[:, ::-1]
        
        if flip_dx:
            voxel_labels = voxel_labels[::-1]
        
        voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat

def custom_rotate_3d(voxel_labels, rotate_degree):
    # rotate like images: convert to PIL Image and rotate
    is_tensor = False
    if type(voxel_labels) is torch.Tensor:
        is_tensor = True
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    voxel_labels_list = []
    for height_index in range(voxel_labels.shape[-1]):
        bev_labels = voxel_labels[..., height_index]
        bev_labels = Image.fromarray(bev_labels.astype(np.uint8))
        bev_labels = bev_labels.rotate(rotate_degree, resample=Image.Resampling.NEAREST, fillcolor=255)
        bev_labels = np.array(bev_labels)
        voxel_labels_list.append(bev_labels)
    voxel_labels = np.stack(voxel_labels_list, axis=-1)
    
    if is_tensor:
        voxel_labels = torch.from_numpy(voxel_labels).long()
    
    return voxel_labels