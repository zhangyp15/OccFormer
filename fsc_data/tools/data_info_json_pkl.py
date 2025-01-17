import pickle
import json
import numpy as np
import os
import re
from copy import deepcopy


#------------生成记录metadata的json文件------------
# 定义文件路径
json_file_template = '/home2/wzc/OccFormer/fsc_data/data/infos/nuscenes_info_0_108.json'
json_file_obj = '/home2/wzc/OccFormer/fsc_data/data/infos/data_info_0_108.json'
pkl_file_obj = '/home2/wzc/OccFormer/fsc_data/data/infos/data_info_0_108.pkl'
image_folder = "/home2/wzc/OccFormer/fsc_data/data/images_3/CAM_FRONT"
image_root_foloder = "/home2/wzc/OccFormer/fsc_data/data/images_3"


# 使用正则表达式从 output_folder 提取相机名称
cam_match = re.search(r"/([^/]+)$", image_folder)
if cam_match:
    cam_name = cam_match.group(1)
    print("相机名称:", cam_name)
else:
    cam_name = "UNKNOWN"
    print("未能从文件路径中提取出相机名称")


# 读取 JSON 文件
with open(json_file_template, 'r') as json_file:
    d = json.load(json_file)

# 获取文件夹中所有文件的路径
# image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
# 获取文件夹中所有以 .jpg 结尾的图片文件路径，并按照时间戳升序排序
image_files = sorted(
    [f for f in os.listdir(image_folder) if f.endswith('.jpg')],
    key=lambda x: int(x.split('_')[3]),  # 提取文件名中的时间戳部分进行排序
)

# 统计图片数量
total_images = len(image_files)

#先把模版中的这个list补够个数，有多少帧图片这个列表就有多少个元素
# 使用 deepcopy 确保每个元素是原始元素的独立副本，必须是deepcopy，要不下面改动一个其他会跟着变
data = {}
# data['infos'] = [deepcopy(d[0]) for _ in range(total_images)]
data['infos'] = d
data['metadata'] = {'version': "fsc"}

cam_intrinsic = [
                 [1.52603682e+03, 0.00000000e+00, 9.68836593e+02],
                 [0.00000000e+00, 1.52379856e+03, 7.30486927e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                ]
scene_token = d[0]['scene_token']
#再挨个对data这个列表中的每个元素更新对应的metadata
for x in range(total_images):
    data['infos'][x]['cams'][cam_name]['data_path'] = image_root_foloder + "/" + cam_name + "/" + image_files[x]
    data['infos'][x]['frame_idx'] = x
    data['infos'][x]['timestamp'] = int(image_files[x][28:44])
    # data['infos'][x]['scene_token'] = 1
    data['infos'][x]['scene_token'] = scene_token
    data['infos'][x]['scene_name'] = "scene-0003"

    data['infos'][x]['cams']['CAM_FRONT']['cam_intrinsic'] = cam_intrinsic
    data['infos'][x]['cams']['CAM_FRONT_RIGHT']['cam_intrinsic'] = cam_intrinsic
    data['infos'][x]['cams']['CAM_FRONT_LEFT']['cam_intrinsic'] = cam_intrinsic
    data['infos'][x]['cams']['CAM_BACK']['cam_intrinsic'] = cam_intrinsic
    data['infos'][x]['cams']['CAM_BACK_LEFT']['cam_intrinsic'] = cam_intrinsic
    data['infos'][x]['cams']['CAM_BACK_RIGHT']['cam_intrinsic'] = cam_intrinsic



# 保存为 JSON 文件: json_file_obj
with open(json_file_obj, 'w') as json_file:
    json.dump(data, json_file, indent=4)


#----------将生成好的json文件保存为.pkl文件-----------
# 定义文件路径
# json_file_path = '/home2/wzc/OccFormer/data_info_pre_test/nuscenes_info_01.json'
# pkl_file_path = '/home2/wzc/OccFormer/data_info_pre_test/nuscenes_info_01.pkl'

# 读取 JSON 文件
with open(json_file_obj, 'r') as json_file:
    data = json.load(json_file)

# 递归函数，将特定的 list 还原为 ndarray
def revert_ndarray(obj):
    if isinstance(obj, list):
        # 检查列表中的每个元素，如果是列表则转换为 ndarray
        return np.array([revert_ndarray(i) for i in obj])
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # 检查指定的键
            if k in {'sweeps', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation'}:
                result[k] = v if isinstance(v, list) else revert_ndarray(v)
            elif k == 'cams' and isinstance(v, dict):
                # 处理 cams 字典中的子元素
                 # 处理 cams 字典中的每个子字典
                result[k] = {
                    cam_key: 
                    {
                        sub_k: sub_v if isinstance(sub_v, list) and sub_k in 
                        {'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation', 'ego2global_rotation'}
                        else revert_ndarray(sub_v)
                        for sub_k, sub_v in cam_val.items()
                    } 
                    for cam_key, cam_val in v.items()
                }
            else:
                # 递归处理其他键
                result[k] = revert_ndarray(v)
        return result
    else:
        return obj

# 转换 JSON 数据中的特定列表为 ndarray
# data_reverted = [revert_ndarray(sample) for sample in data['infos']]
for i, sample in enumerate(data['infos']):
    data['infos'][i] = revert_ndarray(sample)

# 保存为 .pkl 文件
with open(pkl_file_obj, 'wb') as pkl_file:
    pickle.dump(data, pkl_file)

print(f"Data has been saved to {pkl_file_obj}")