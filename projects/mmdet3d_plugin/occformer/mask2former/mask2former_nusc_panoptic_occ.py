# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Conv3d, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import (get_nusc_lidarseg_point_coords, 
                        preprocess_panoptic_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from projects.mmdet3d_plugin.utils import per_class_iu, fast_hist_crop, PanopticEval

import pdb

# Mask2former for 3D Occupancy Segmentation on nuScenes dataset
@HEADS.register_module()
class Mask2FormerNuscPanopticOccHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 point_cloud_range=None,
                 padding_mode='border',
                 stuff_indices=None,
                 thing_indices=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        self.point_cloud_range = point_cloud_range
        
        self.stuff_indices = stuff_indices
        self.thing_indices = thing_indices
        
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution, align the channel of input features
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv3d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        ''' Pixel Decoder Related, skipped '''
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.pooling_attn_mask = pooling_attn_mask
        self.class_weight = loss_cls.class_weight
        
        # align_corners
        self.align_corners = True
        self.padding_mode = padding_mode
        
        # panoptic metric
        self.panoptic_metric = PanopticEval(self.num_classes, ignore=[0], min_points=15)
        
    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)
        
        if hasattr(self, "pixel_decoder"):
            self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, gt_lidarseg_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, mask_preds_list, 
                    gt_labels_list, gt_masks_list, gt_lidarseg_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, 
                    gt_lidarseg, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        
        point_cloud_range = torch.tensor(self.point_cloud_range).type_as(gt_lidarseg)
        point_coords = (gt_lidarseg[:, :3] - point_cloud_range[:3]) / (point_cloud_range[3:] - point_cloud_range[:3])
        
        num_lidarseg = min(self.num_points // 2, point_coords.shape[0])
        if num_lidarseg < point_coords.shape[0]:
            point_coords = point_coords[torch.randperm(point_coords.shape[0])[:num_lidarseg]]
        
        num_rand = self.num_points - num_lidarseg
        rand_point_coords = torch.rand((num_rand, 3), device=cls_score.device)
        point_coords = torch.cat((point_coords, rand_point_coords), dim=0)
        point_coords = point_coords[..., [2, 1, 0]]
        
        # since there are out-of-range lidar points, the padding_mode is set to border
        mask_points_pred = point_sample_3d(mask_pred.unsqueeze(1), 
            point_coords.repeat(num_queries, 1, 1), padding_mode=self.padding_mode).squeeze(1)
        
        gt_points_masks = point_sample_3d(gt_masks.unsqueeze(1).float(), 
            point_coords.repeat(num_gts, 1, 1), padding_mode=self.padding_mode).squeeze(1)
        
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]

        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, gt_lidarseg_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_lidarseg_list = [gt_lidarseg_list for _ in range(num_dec_layers)]
        
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, all_gt_lidarseg_list, img_metas_list)
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, gt_lidarseg_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, x, y, z).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, x, y, z).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, 
                    gt_masks_list, gt_lidarseg_list, img_metas)
        
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_weights = mask_weights[mask_weights > 0]
        
        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        
        with torch.no_grad():
            point_coords = get_nusc_lidarseg_point_coords(mask_preds.unsqueeze(1), 
                gt_lidarseg_list, gt_labels_list, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio, self.point_cloud_range, 
                padding_mode=self.padding_mode)
        
        point_coords = point_coords[..., [2, 1, 0]]
        mask_point_preds = point_sample_3d(mask_preds.unsqueeze(1), 
            point_coords, padding_mode=self.padding_mode).squeeze(1)
        
        # dice loss
        mask_point_targets = point_sample_3d(mask_targets.unsqueeze(1).float(), 
            point_coords, padding_mode=self.padding_mode).squeeze(1)
        
        # the weighted version
        num_total_mask_weights = reduce_mean(mask_weights.sum())
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                        weight=mask_weights, avg_factor=num_total_mask_weights)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)

        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_mask_weights * self.num_points,
        )

        return loss_cls, loss_mask, loss_dice

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries, x, y, z).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)
        
        ''' 对于一些样本数量较少的类别来说，经过 trilinear 插值 + 0.5 阈值，正样本直接消失 '''
        
        if self.pooling_attn_mask:
            # however, using max-pooling can save more positive samples, which is quite important for rare classes
            attn_mask = F.adaptive_max_pool3d(mask_pred.float(), attn_mask_target_size)
        else:
            # by default, we use trilinear interp for downsampling
            attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='trilinear', align_corners=self.align_corners)
        
        # merge the dims of [x, y, z]
        attn_mask = attn_mask.flatten(2).detach() # detach the gradients back to mask_pred
        attn_mask = attn_mask.sigmoid() < 0.5
        
        # repeat for the num_head axis, (batch_size, num_queries, num_seq) -> (batch_size * num_head, num_queries, num_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)

        return cls_pred, mask_pred, attn_mask

    def preprocess_gt(self, gt_occ, img_metas):
        
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        num_query_list = [self.num_queries] * len(img_metas)
        targets = multi_apply(preprocess_panoptic_occupancy_gt, gt_occ, num_class_list, img_metas, num_query_list)
        
        labels, masks = targets
        return labels, masks

    def forward_lidar_panopticseg(self, cls_preds, mask_preds, points, img_metas=None):
        pc_range = torch.tensor(img_metas[0]['pc_range']).type_as(mask_preds)
        pc_range_min = pc_range[:3]
        pc_range = pc_range[3:] - pc_range_min
        
        # sample the corresponding mask_preds
        point_mask_preds = []
        for batch_index, points_i in enumerate(points):
            points_i = (points_i[:, :3].float() - pc_range_min) / pc_range
            points_i = (points_i * 2) - 1
            points_i = points_i[..., [2, 1, 0]]
            points_i = points_i.view(1, 1, 1, -1, 3)
            
            point_mask_preds_i = F.grid_sample(mask_preds[batch_index : batch_index + 1], points_i, 
                    mode='bilinear', padding_mode=self.padding_mode, align_corners=self.align_corners)
            # (num_query, num_points)
            point_mask_preds_i = point_mask_preds_i.squeeze().contiguous()
            point_mask_preds.append(point_mask_preds_i)
        
        # this line only supports batch_size = 1
        point_mask_preds = torch.stack(point_mask_preds, dim=0)
        point_semantic_preds, point_panoptic_preds = self.format_panoptic_lidar_results(cls_preds, point_mask_preds)
        
        if self.training:
            # compute panoptic metrics
            point_panoptic_labels = torch.cat([x[:, -1] for x in points]).long()
            point_semantic_labels = point_panoptic_labels // 1e3
            
            point_panoptic_labels = point_panoptic_labels.cpu().numpy()
            point_semantic_labels = point_semantic_labels.cpu().numpy()
            point_panoptic_preds = point_panoptic_preds.flatten().cpu().numpy()
            point_semantic_preds = point_semantic_preds.flatten().cpu().numpy()
            
            PQ, SQ, RQ = self.panoptic_metric.evaluate_panoptic_single(point_semantic_preds, point_panoptic_preds, point_semantic_labels, point_panoptic_labels)
            
            metric_dict = {
                'point_panoptic_pq': torch.tensor(PQ).cuda(), 
                'point_panoptic_sq': torch.tensor(SQ).cuda(), 
                'point_panoptic_rq': torch.tensor(RQ).cuda()}
            
            # compute semantic metrics
            unique_label = np.arange(16)
            hist = fast_hist_crop(point_semantic_preds, point_semantic_labels, unique_label)
            iou = per_class_iu(hist)
            metric_dict['point_seg_miou'] = torch.tensor(np.nanmean(iou)).cuda()
            
            return metric_dict
        else:
            return point_semantic_preds, point_panoptic_preds
    
    def forward_train(self,
            voxel_feats,
            img_metas,
            gt_occ,
            points=None,
            **kwargs,
        ):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # forward
        all_cls_scores, all_mask_preds = self(voxel_feats, img_metas)
        
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_occ, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, points, img_metas)
        
        # forward_lidarseg
        with torch.no_grad():
            losses_lidarseg = self.forward_lidar_panopticseg(all_cls_scores[-1], 
                            all_mask_preds[-1], points, img_metas)
        
        losses.update(losses_lidarseg)

        return losses

    def forward(self, 
            voxel_feats,
            img_metas,
            **kwargs,
        ):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 5D-tensor (B, C, X, Y, Z).
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        """
        
        batch_size = len(img_metas)
        mask_features = voxel_feats[0]
        multi_scale_memorys = voxel_feats[:0:-1]
        
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            ''' with flatten features '''
            # projection for input features
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            ''' with level embeddings '''
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            ''' with positional encodings '''
            # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-3:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
            
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        
        ''' directly deocde the learnable queries, as simple proposals '''
        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, 
                    mask_features, multi_scale_memorys[0].shape[-3:])
    
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                key_padding_mask=None)
            
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, 
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-3:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        
        '''
        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        '''
        
        return cls_pred_list, mask_pred_list

    def format_panoptic_lidar_results(self, mask_cls_results, mask_pred_results):
        point_semantic_preds_list = []
        point_panoptic_preds_list = []
        for batch_index in range(mask_cls_results.shape[0]):
            mask_cls = mask_cls_results[batch_index]
            mask_pred = mask_pred_results[batch_index]
            
            # for each query, find its foreground labels
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            labels = mask_cls[..., 1:].argmax(-1)
            labels += 1
            
            point_indices = mask_pred.argmax(dim=0)
            point_semantic_preds = labels[point_indices]
            point_panoptic_preds = point_semantic_preds.new_zeros(point_semantic_preds.shape).long()
            
            instance_id = 1
            for label_id in torch.unique(point_semantic_preds):
                label_mask = (point_semantic_preds == label_id)
                if int(label_id.item()) not in self.thing_indices:
                    point_panoptic_preds[label_mask] = (label_id * 1e3).long()
                    continue
                
                label_query_indices = point_indices[label_mask]
                for query_index in torch.unique(label_query_indices):
                    query_mask = (point_indices == query_index)
                    point_panoptic_preds[query_mask] = (label_id * 1e3 + instance_id).long()
                    instance_id += 1
            
            point_semantic_preds_list.append(point_semantic_preds)
            point_panoptic_preds_list.append(point_panoptic_preds)
        
        return torch.stack(point_semantic_preds_list), torch.stack(point_panoptic_preds_list)
    
    def format_panoptic_occupancy_results(self, mask_cls_results, mask_pred_results):
        voxel_semantic_preds_list = []
        voxel_panoptic_preds_list = []
        for batch_index in range(mask_cls_results.shape[0]):
            mask_cls = mask_cls_results[batch_index]
            mask_pred = mask_pred_results[batch_index]
            
            # for each query, find its corresponding labels
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            scores, labels = mask_cls.max(-1)
            
            # the query index, which each voxel is matched with
            voxel_indices = mask_pred.argmax(dim=0)
            voxel_semantic_preds = labels[voxel_indices]
            voxel_panoptic_preds = voxel_semantic_preds.new_zeros(voxel_semantic_preds.shape).long()
            
            instance_id = 1
            for label_id in torch.unique(voxel_semantic_preds):
                # for each valid semantic class, find its corresponding voxel positions
                label_mask = (voxel_semantic_preds == label_id)
                # if the class is stuff_class, simply treat it as semantic segmentation
                if int(label_id.item()) not in self.thing_indices:
                    voxel_panoptic_preds[label_mask] = (label_id * 1e3).long()
                    continue
                
                # else: treat it as panoptic segmentation, find all query instances
                label_query_indices = voxel_indices[label_mask]
                for query_index in torch.unique(label_query_indices):
                    query_mask = (voxel_indices == query_index)
                    voxel_panoptic_preds[query_mask] = (label_id * 1e3 + instance_id).long()
                    instance_id += 1
            
            voxel_semantic_preds_list.append(voxel_semantic_preds)
            voxel_panoptic_preds_list.append(voxel_panoptic_preds)
        
        return torch.stack(voxel_semantic_preds_list), torch.stack(voxel_panoptic_preds_list)

    def simple_test(self, 
            voxel_feats,
            img_metas,
            points=None,
            **kwargs,
        ):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        all_cls_scores, all_mask_preds = self(voxel_feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        
        # rescale mask prediction
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=tuple(img_metas[0]['occ_size']),
            mode='trilinear',
            align_corners=self.align_corners,
        )
        
        output_voxels = self.format_panoptic_occupancy_results(mask_cls_results, mask_pred_results)
        res = {
            'output_voxels': [output_voxels],
            'output_points': None,
        }
        
        point_semantic_preds, point_panoptic_preds = self.forward_lidar_panopticseg(
            cls_preds=all_cls_scores[-1],
            mask_preds=all_mask_preds[-1],
            points=points,
            img_metas=img_metas,
        )
        res['output_points'] = point_semantic_preds
        res['output_panoptic_points'] = point_panoptic_preds

        return res
