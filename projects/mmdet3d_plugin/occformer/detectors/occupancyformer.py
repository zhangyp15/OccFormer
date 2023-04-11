import torch
import torch.nn.functional as F
import collections 

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth

import numpy as np
import time
import pdb

@DETECTORS.register_module()
class OccupancyFormer(BEVDepth):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    @force_fp32()
    def bev_encoder(self, x):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        x = self.img_bev_encoder_backbone(x)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['bev_encoder'].append(t1 - t0)
        
        x = self.img_bev_encoder_neck(x)
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['bev_neck'].append(t2 - t1)
        
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        x = self.image_encoder(img[0])
        img_feats = x.clone()
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        x = self.bev_encoder(x)
        if type(x) is not list:
            x = [x]
        
        return x, depth, img_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (voxel_feats, img_feats, depth)
    
    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(
            self,
            pts_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            img_feats=None,
            points_uv=None,
            **kwargs,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        losses = self.pts_bbox_head.forward_train(
            voxel_feats=pts_feats,
            img_metas=img_metas,
            gt_occ=gt_occ,
            points=points_occ,
            img_feats=img_feats,
            points_uv=points_uv,
            **kwargs,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['mask2former_head'].append(t1 - t0)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[7], depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ, 
                        points_occ, img_metas, img_feats=img_feats, points_uv=points_uv, **kwargs)
        losses.update(losses_occupancy)
        
        if self.record_time:
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
                
            print(out_res)
        
        return losses
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, **kwargs)
    
    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, gt_occ=None, points_uv=None):
        voxel_feats, img_feats, depth = self.extract_feat(points=None, img=img, img_metas=img_metas)        
        output = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
        )
        
        # evaluate nusc lidar-seg
        if output['output_points'] is not None and points_occ is not None:
            output['output_points'] = torch.argmax(output['output_points'][:, 1:], dim=1) + 1
            target_points = torch.cat(points_occ, dim=0)
            output['evaluation_semantic'] = self.simple_evaluation_semantic(output['output_points'], target_points, img_metas)
            output['target_points'] = target_points
        
        # evaluate voxel 
        output_voxels = output['output_voxels'][0]
        target_occ_size = img_metas[0]['occ_size']
        
        if (output_voxels.shape[-3:] != target_occ_size).any():
            output_voxels = F.interpolate(output_voxels, size=tuple(target_occ_size), 
                            mode='trilinear', align_corners=True)
        
        output['output_voxels'] = output_voxels
        output['target_voxels'] = gt_occ
        
        return output

    def post_process_semantic(self, pred_occ):
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        score, clses = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        return clses

    def simple_evaluation_semantic(self, pred, gt, img_metas):
        # pred = torch.argmax(pred, dim=1).cpu().numpy()
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        gt = gt[:, 3].astype(np.int)
        unique_label = np.arange(16)
        hist = fast_hist_crop(pred, gt, unique_label)
        
        return hist

@DETECTORS.register_module()
class OccupancyFormer4D(OccupancyFormer):
    def prepare_voxel_feat(self, img, rot, tran, intrin, 
                post_rot, post_tran, bda, mlp_input):
        
        x = self.image_encoder(img)
        img_feats = x.clone()
        
        voxel_feat, depth = self.img_view_transformer([x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        
        return voxel_feat, depth, img_feats

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        voxel_feat_list = []
        img_feat_list = []
        depth_list = []
        key_frame = True # back propagation for key frame only
        
        for img, rot, tran, intrin, post_rot, \
            post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
                
            mlp_input = self.img_view_transformer.get_mlp_input(
                rots[0], trans[0], intrin,post_rot, post_tran, bda)
            inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda, mlp_input)
            if not key_frame:
                with torch.no_grad():
                    voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            else:
                voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            
            voxel_feat_list.append(voxel_feat)
            img_feat_list.append(img_feats)
            depth_list.append(depth)
            key_frame = False
        
        voxel_feat = torch.cat(voxel_feat_list, dim=1)
        x = self.bev_encoder(voxel_feat)
        if type(x) is not list:
            x = [x]

        return x, depth_list[0], img_feat_list[0]
        