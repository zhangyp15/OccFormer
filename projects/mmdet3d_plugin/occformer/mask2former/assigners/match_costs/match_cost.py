# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class DiceCost:
    """Cost of mask assignments based on dice losses.

    Args:
        weight (int | float, optional): loss_weight. Defaults to 1.
        pred_act (bool, optional): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float, optional): default 1e-12.
        naive_dice (bool, optional): If True, use the naive dice loss
            in which the power of the number in the denominator is
            the first power. If Flase, use the second power that
            is adopted by K-Net and SOLO.
            Defaults to True.
    """

    def __init__(self, weight=1., pred_act=False, eps=1e-3, naive_dice=True):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def binary_mask_dice_loss(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (num_query, *).
            gt_masks (Tensor): Ground truth in shape (num_gt, *)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (num_query, num_gt).
        """
        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + \
                gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + \
                gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction logits in shape (num_query, *)
            gt_masks (Tensor): Ground truth in shape (num_gt, *)

        Returns:
            Tensor: Dice cost matrix with weight in shape (num_query, num_gt).
        """
        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight

@MATCH_COST.register_module()
class CrossEntropyLossCost:
    """CrossEntropyLossCost.

    Args:
        weight (int | float, optional): loss weight. Defaults to 1.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
    Examples:
         >>> from mmdet.core.bbox.match_costs import CrossEntropyLossCost
         >>> import torch
         >>> bce = CrossEntropyLossCost(use_sigmoid=True)
         >>> cls_pred = torch.tensor([[7.6, 1.2], [-1.3, 10]])
         >>> gt_labels = torch.tensor([[1, 1], [1, 0]])
         >>> print(bce(cls_pred, gt_labels))
    """

    def __init__(self, weight=1., use_sigmoid=True):
        assert use_sigmoid, 'use_sigmoid = False is not supported yet.'
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).

        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
            gt_labels (Tensor): Labels.

        Returns:
            Tensor: Cross entropy cost matrix with weight in
                shape (num_query, num_gt).
        """
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight