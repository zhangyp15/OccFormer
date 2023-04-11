# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import time
import os

import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger

import mmcv
import numpy as np
from fvcore.nn import parameter_count_table
from projects.mmdet3d_plugin.utils import cm_to_ious, format_results, SSCMetrics

# utils for saving predictions 
from .utils import *

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3, pred_save=None, test_save=None):
    model.eval()
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    # evaluate ssc
    is_semkitti = hasattr(dataset, 'camera_used')
    ssc_metric = SSCMetrics().cuda()
    logger.info(parameter_count_table(model, max_depth=4))
    
    batch_size = 1
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
            
            # for one-gpu test, print results for each batch
            ious = cm_to_ious(evaluation_semantic)
            res_table, _ = format_results(ious, return_dic=True)
            print(res_table)
        
        img_metas = data['img_metas'].data[0][0]
        # save for test submission
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            target_voxels = result['target_voxels'].clone()
            ssc_metric.update(y_pred=output_voxels,  y_true=target_voxels)
            
            # compute metrics
            scores = ssc_metric.compute()
            if is_semkitti:
                print('\n Evaluating semanticKITTI occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            else:
                print('\n Evaluating nuScenes occupancy: SC IoU = {:.3f}, SSC mIoU = {:.3f}'.format(scores['iou'], 
                                    scores['iou_ssc_mean']))
            
            # save for val predictions, mostly for visualization
            if pred_save is not None:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'], 
                        target_points=result['target_points'], 
                        save_path=pred_save, 
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))
        
        for _ in range(batch_size):
            prog_bar.update()
    
    res = {
        'ssc_scores': ssc_metric.compute(),
    }
    
    if type(evaluation_semantic) is np.ndarray:
        res['evaluation_semantic'] = evaluation_semantic
    
    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, pred_save=None, test_save=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    ssc_results = []
    ssc_metric = SSCMetrics().cuda()
    is_semkitti = hasattr(dataset, 'camera_used')
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    is_val_save_predictins = pred_save is not None
    if is_val_save_predictins:
        os.makedirs(pred_save, exist_ok=True)
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    batch_size = 1
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
        
        img_metas = data['img_metas'].data[0][0]
        # occupancy prediction
        if is_test_submission:
            if is_semkitti:
                assert result['output_voxels'].shape[0] == 1
                save_output_semantic_kitti(result['output_voxels'][0], 
                    test_save, img_metas['sequence'], img_metas['frame_id'])
            else:
                save_nuscenes_lidarseg_submission(result['output_points'], test_save, img_metas)
        else:
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            
            if result['target_voxels'] is not None:
                target_voxels = result['target_voxels'].clone()
                ssc_results_i = ssc_metric.compute_single(
                    y_pred=output_voxels, y_true=target_voxels)
                ssc_results.append(ssc_results_i)
            
            if is_val_save_predictins:
                if is_semkitti:
                    save_output_semantic_kitti(result['output_voxels'][0], pred_save, 
                        img_metas['sequence'], img_metas['frame_id'], raw_img=img_metas['raw_img'], test_mapping=False)
                
                else:
                    save_output_nuscenes(data['img_inputs'], output_voxels, 
                        output_points=result['output_points'],
                        target_points=result['target_points'], 
                        save_path=pred_save,
                        scene_token=img_metas['scene_token'], 
                        sample_token=img_metas['sample_idx'],
                        img_filenames=img_metas['img_filenames'],
                        timestamp=img_metas['timestamp'],
                        scene_name=img_metas.get('scene_name', None))
        
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    # wait until all predictions are generated
    dist.barrier()
    
    if is_test_submission:
        return None
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    
    if type(evaluation_semantic) is np.ndarray:
        # convert to tensor for reduce_sum
        evaluation_semantic = torch.from_numpy(evaluation_semantic).cuda()
        dist.all_reduce(evaluation_semantic, op=dist.ReduceOp.SUM)
        res['evaluation_semantic'] = evaluation_semantic.cpu().numpy()
    
    return res

