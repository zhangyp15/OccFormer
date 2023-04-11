import numpy as np
import yaml, os
import torch
import numba as nb

from mmdet.datasets.builder import PIPELINES
from .loading_nusc_occ import voxel_transform

import pdb

'''
Instructions from PanopticNuScenes:

A ground truth label file named {token}_panoptic.npz is provided for each sample in the Panoptic nuScenes dataset. 
A .npz file contains the panoptic label array (uint16 format) of the corresponding points in a pointcloud. 
The panoptic label of each point is: (general class index * 1000 + instance index).
Note here general class index (32 classes in total) rather than the challenge class index (16 classes in total) is used. 
For example, a ground truth instance from car class (general class index = 17), and with assigned car instance index 1, 
    will have a ground truth panoptic label of 1000 * 17 + 1 = 17001 in the .npz file. 
Since these ground truth panoptic labels are generated from annotated bounding boxes, 
    points that are included in more than 1 bounding box will be ignored, and assigned with panoptic label 0: 
    class index 0 and instance index 0. For points from stuff, their panoptic labels will be general class index 1000. 

To align with thing classes, you may think the stuff classes as sharing an instance index of 0 by all points. 
To load a ground truth file, you can use:
    from nuscenes.utils.data_io import load_bin_file
    label_file_path = /data/sets/nuscenes/panoptic/v1.0-mini/{token}_panoptic.npz
    panoptic_label_arr = load_bin_file(label_file_path, 'panoptic')

'''

@PIPELINES.register_module()
class LoadNuscPanopticOccupancyAnnotations(object):
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
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def __call__(self, results):
        # for test-submission of nuScenes Panoptic Segmentation 
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
        
        panopticseg_file = results['lidarseg'].replace('lidarseg', 'panoptic')
        panopticseg_file = panopticseg_file.replace('.bin', '.npz')
        panopticseg_file = os.path.join(self.data_root, panopticseg_file)
        panoptic_labels = np.load(panopticseg_file)['data']
        semantic_labels = panoptic_labels // 1e3
        
        # perform learning mapping
        for semantic_label in np.unique(semantic_labels):
            semantic_mask = (semantic_labels == semantic_label)
            mapped_id = self.learning_map[int(semantic_label)]
            semantic_labels[semantic_mask] = mapped_id
            
            cls_panoptic_labels = panoptic_labels[semantic_mask] - semantic_label * 1e3
            panoptic_labels[semantic_mask] = mapped_id * 1e3 + cls_panoptic_labels
        
        panoptic_labels = panoptic_labels.reshape(-1, 1)
        points = np.fromfile(results['pts_filename'], dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
        lidarseg = np.concatenate([points, panoptic_labels], axis=-1)
        
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
        label_voxel_pair = label_voxel_pair.astype(np.uint16)
        
        # 0: noise, 1-16 normal classes, 17 unoccupied
        empty_id = int(self.unoccupied_id * 1e3)
        processed_label = np.ones(self.grid_size, dtype=np.uint16) * empty_id
        processed_label = nb_process_label(processed_label, label_voxel_pair)
        
        # convert label_0 to label_65535 (ignored)
        processed_label[processed_label == 0] = 65535
        # convert empty to label id 0
        processed_label[processed_label == empty_id] = 0
        
        ''' save results'''
        results['gt_occ'] = torch.from_numpy(processed_label.astype(np.float32)).long()
        results['points_occ'] = torch.from_numpy(lidarseg).float()
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)

        return results
    
@nb.jit('u2[:,:,:](u2[:,:,:], u2[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    ignore_index = 0
    label_size = 256 * 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            counter[ignore_index] = 0
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        
        counter[sorted_label_voxel_pair[i, 3]] += 1
    
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label