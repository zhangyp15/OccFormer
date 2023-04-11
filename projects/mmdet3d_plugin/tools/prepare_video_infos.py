import pickle
import argparse
import os
import numpy as np

from copy import deepcopy
from nuscenes import NuScenes
from collections import defaultdict
from tqdm import tqdm

import pdb

def arange_according_to_scene(infos, nusc):
    scenes = defaultdict(list)
    print('rearrange infos according to scenes ...')
    for i, info in tqdm(enumerate(infos), total=len(infos)):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']
        scenes[scene_name].append(info)
    
    return scenes

def insert_sweeps(data_infos, data_path):
    camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    camera_to_index = {camera_name : index for index, camera_name in enumerate(camera_names)}
    
    # load sweep names & timestamps
    sweep_cams = []
    sweep_tss = []
    for cam_type in camera_names:
        dir = os.path.join(data_path, 'sweeps', cam_type)
        filenames = os.listdir(dir)
        files = [os.path.join(dir, fn) for fn in filenames]
        ts = [int(fn.split('__')[-1].split('.')[0]) for fn in filenames]
        idx = np.argsort(ts)
        sweep_cams.append(np.array(files)[idx])
        sweep_tss.append(np.array(ts)[idx])
    
    sweep_cams = np.array(sweep_cams)
    sweep_tss = np.array(sweep_tss)
    
    print('insert sweep infos ...')
    output_infos = []
    for scene_name, scene_infos in tqdm(data_infos.items(), total=len(data_infos)):
        for index in range(len(scene_infos) - 1):
            # search corresponding sweeps
            start_ts = scene_infos[index]['timestamp']
            end_ts = scene_infos[index + 1]['timestamp']
            temp_cams = []
            
            for sweep_cam, sweep_ts in zip(sweep_cams, sweep_tss):
                insert_indices = np.nonzero((sweep_ts > start_ts) & (sweep_ts < end_ts))[0]
                temp_cam = sweep_cam[insert_indices]
                temp_cams.append(temp_cam.tolist())
            
            # camera_list, sweep_list
            min_len = min([len(temp_cam) for temp_cam in temp_cams])
            temp_cams = [temp_cam[:min_len] for temp_cam in temp_cams]
            
            ''' 
            scene_info includes the following:
                dict_keys(['lidar_path', 'token', 'prev', 'next', 'can_bus', 'frame_idx', 'sweeps', 'cams', 'scene_token', 
                'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 
                'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag'])
            '''
            
            # add interbetween sweeps
            scene_infos[index]['scene_name'] = scene_name
            insert_items = []
            for j in range(min_len):
                temp_dict = deepcopy(scene_infos[index])
                # replace image path
                for cam_type, cam_info in temp_dict['cams'].items():
                    cam_info['data_path'] = temp_cams[camera_to_index[cam_type]][j]
                # replace timestamp with the front-camera timestamp
                temp_dict['timestamp'] = int(temp_cams[0][j].split('__')[-1].split('.')[0])
                insert_items.append(temp_dict)
            
            output_infos.append(scene_infos[index])
            output_infos.extend(insert_items)
        
        # append the last sample for each scene
        scene_infos[-1]['scene_name'] = scene_name
        output_infos.append(scene_infos[-1])
        
        # 2Hz to 12Hz, around 6x
        print('finish processing scene {}, num sample {}'.format(scene_name, len(output_infos)))

    return output_infos

if __name__ == "__main__":
    parse = argparse.ArgumentParser('')
    parse.add_argument('--src-path', type=str, default='', help='path of the original pkl file')
    parse.add_argument('--dst-path', type=str, default='', help='path of the output pkl file')
    parse.add_argument('--data-path', type=str, default='', help='path of the nuScenes dataset')
    args = parse.parse_args()

    with open(args.src_path, 'rb') as f:
        data = pickle.load(f)
    nusc = NuScenes('v1.0-trainval', args.data_path)
    data['infos'] = arange_according_to_scene(data['infos'], nusc)
    data['infos'] = insert_sweeps(data['infos'], args.data_path)
    
    with open(args.dst_path, 'wb') as f:
        pickle.dump(data, f)
    