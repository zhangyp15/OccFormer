import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset

import pdb

@DATASETS.register_module()
class CustomNuScenesOccLSSDataset(NuScenesDataset):
    def __init__(self, occ_size, pc_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.occ_size = occ_size
        self.pc_range = pc_range
        
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            # fix for running the code on different data_paths
            pts_filename=info['lidar_path'].replace('./data/nuscenes', self.data_root),
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            lidar_token=info['lidar_token'],
        )
        
        # available for infos which are organized in scenes and prepared for video demos
        if 'scene_name' in info:
            input_dict['scene_name'] = info['scene_name']
        
        # not available for test-test
        if 'lidarseg' in info:
            input_dict['lidarseg'] = info['lidarseg']
        
        # fix data_path
        img_filenames = {}
        lidar2cam_dic = {}
        
        for cam_type, cam_info in info['cams'].items():
            cam_info['data_path'] = cam_info['data_path'].replace('./data/nuscenes', self.data_root)
            img_filenames[cam_type] = cam_info['data_path']
            
            # obtain lidar to camera transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_dic[cam_type] = lidar2cam_rt.T
        
        input_dict['curr'] = info
        input_dict['img_filenames'] = img_filenames
        input_dict['lidar2cam_dic'] = lidar2cam_dic
        
        return input_dict

    def evaluate_lidarseg(self, results, logger=None, **kwargs):
        from projects.mmdet3d_plugin.utils import cm_to_ious, format_results
        eval_results = {}
        
        ''' evaluate lidar semantic segmentation '''
        ious = cm_to_ious(results['evaluation_semantic'])
        res_table, res_dic = format_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['nuScenes_lidarseg_{}'.format(key)] = val
        
        if logger is not None:
            logger.info('LiDAR Segmentation Evaluation')
            logger.info(res_table)
        
        return eval_results

    def evaluate_ssc(self, results, logger=None, **kwargs):
        # though supported, it can only be evaluated by the sparse-LiDAR-generated occupancy in nusc
        
        eval_results = {}
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']
            
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
        
        class_names = ['empty', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 
                       'pedestrian',  'traffic_cone', 'trailer', 'truck', 'driveable_surface', 
                       'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
        
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        for key, val in res_dic.items():
            eval_results['nuScenes_{}'.format(key)] = round(val * 100, 2)
        
        # add two main metrics to serve as the sort metric
        eval_results['nuScenes_combined_IoU'] = eval_results['nuScenes_SC_IoU'] + eval_results['nuScenes_SSC_mIoU']
        
        if logger is not None:
            logger.info('NuScenes SSC Evaluation')
            logger.info(eval_results)

    def evaluate(self, results, logger=None, **kwargs):
        if results is None:
            logger.info('Skip Evaluation')
        
        if 'evaluation_semantic' in results:
            return self.evaluate_lidarseg(results, logger, **kwargs)
        else:
            return self.evaluate_ssc(results, logger, **kwargs)
        