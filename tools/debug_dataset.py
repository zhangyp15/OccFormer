# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
from tqdm import tqdm, trange
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import pdb
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--test', action='store_true',)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    if args.test:
        datasets = build_dataset(cfg.data.test)
    else:
        datasets = build_dataset(cfg.data.train)
    
    # train_dataset = build_dataset(cfg.data.train)
    # val_dataset = build_dataset(cfg.data.test)
    # cfg.data.test['split'] = 'test-submit'
    # test_data = build_dataset(cfg.data.test)
    
    data_loaders = build_dataloader(
        datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False,
        # shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
        # nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
    )
    
    # datasets.compute_occupancy_class_metas(scale='small')
    # datasets.compute_lidarseg_class_metas()
    # datasets.compute_lidarseg_occupancy_class_metas()
    
    # class_frequencies = np.zeros(17)    
    # for batch in tqdm(data_loaders):
    #     gt_occ = batch['gt_occ'].data[0]
    #     cls_ids, cls_counts = torch.unique(gt_occ, return_counts=True)
        
    #     pdb.set_trace()
        
        # for cls_id, cls_count in zip(cls_ids, cls_counts):
        #     if cls_id == 255:
        #         continue
        #     class_frequencies[cls_id] += cls_count

    # print(class_frequencies.tolist())
    # return 0
    
    shuffle = True
    print('number of data samples = {}'.format(len(datasets)))
    
    # shuffle the dataset
    traverse_indices = list(range(len(datasets)))
    if shuffle:
        import random
        random.shuffle(traverse_indices)
    
    for index in tqdm(traverse_indices):
        sample = datasets[index]
        
if __name__ == '__main__':
    main()
