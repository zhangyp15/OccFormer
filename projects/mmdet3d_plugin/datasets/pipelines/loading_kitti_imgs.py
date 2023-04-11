# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES

import torch
from PIL import Image
from .loading_nusc_imgs import mmlabNormalize

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_SemanticKitti(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False, img_norm_cfg=None):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.img_norm_cfg = img_norm_cfg

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self, results, flip=None, scale=None):
        # load the monocular image for semantic kitti
        img_filenames = results['img_filename']
        assert len(img_filenames) == 1
        img_filenames = img_filenames[0]
        
        img = mmcv.imread(img_filenames, 'unchanged')
        results['raw_img'] = img
        img = Image.fromarray(img)
        
        # perform image-view augmentation
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        
        img_augs = self.sample_augmentation(H=img.height, W=img.width, 
                        flip=flip, scale=scale)

        resize, resize_dims, crop, flip, rotate = img_augs
        img, post_rot2, post_tran2 = \
            self.img_transform(img, post_rot, post_tran, resize=resize, 
                resize_dims=resize_dims, crop=crop,flip=flip, rotate=rotate)

        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        
        # intrins
        intrin = torch.Tensor(results['cam_intrinsic'][0])
        
        # extrins
        lidar2cam = torch.Tensor(results['lidar2cam'][0])
        cam2lidar = lidar2cam.inverse()
        rot = cam2lidar[:3, :3]
        tran = cam2lidar[:3, 3]
        
        results['canvas'] = np.array(img)[None]
        
        img = self.normalize_img(img, img_norm_cfg=self.img_norm_cfg)
        depth = torch.zeros(1)
        
        res = [img, rot, tran, intrin, post_rot, post_tran, depth, cam2lidar]
        res = [x[None] for x in res]
        
        return tuple(res)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        
        return results