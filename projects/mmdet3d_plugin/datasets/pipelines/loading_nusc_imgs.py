# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet.datasets.builder import PIPELINES
import torch
import mmcv
from PIL import Image
import pdb

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_OccFormer(object):
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

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cam_names = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

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
            resize = float(fW)/float(W)
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
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        gt_depths = []
        sensor2sensors = []
        
        canvas = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            
            img = mmcv.imread(filename, 'unchanged')
            img = Image.fromarray(img)
            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            
            # from camera to lidar 
            sensor2lidar = torch.tensor(results['lidar2cam_dic'][cam_name]).inverse().float()
            
            rot = sensor2lidar[:3, :3]
            tran = sensor2lidar[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(H=img.height,
                                                W=img.width,
                                                flip=flip,
                                                scale=scale)
            
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            
            # raw images for visualize
            canvas.append(np.array(img))
                        
            imgs.append(self.normalize_img(img, img_norm_cfg=self.img_norm_cfg))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            gt_depths.append(torch.zeros(1))
            # only placeholder currently, to be used for video-based methods
            sensor2sensors.append(sensor2lidar)
        
        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        gt_depths = torch.stack(gt_depths)
        sensor2sensors = torch.stack(sensor2sensors)
        
        # the RGB uint8 input images, for debug or visualization
        results['canvas'] = np.stack(canvas)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        
        return results

def mmlabNormalize(img, img_norm_cfg=None):
    from mmcv.image.photometric import imnormalize
    if img_norm_cfg is None:
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
    else:
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        to_rgb = img_norm_cfg['to_rgb']
    
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    
    return img

def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
    """Transform depth based on ida augmentation configuration.

    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """

    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)