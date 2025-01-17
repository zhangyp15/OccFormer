import pickle
import json
import numpy as np

#------------------------读取查看.pkl文件----------------------
# 定义文件路径
file_path = '/home2/wzc/OccFormer/prediction/scene-0003/1533201470448696.pkl'

# 读取 .pkl 文件
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Check if 17 is in the array
contains_17 = np.any(data['pred_voxels'] == 17)

print(f"Contains 17: {contains_17}")

# 打印读取的数据
print(data['infos'][0])

# ------------------------读取.pkl文件，并把data['infos'][0]和[1]保存为.json文件-------------------------
# # 定义文件路径
# pkl_file_path = '/home2/wzc/OccFormer/data_info_pre_test/nuscenes_infos_temporal_val.pkl'
# json_file_path = '/home2/wzc/OccFormer/data_info_pre_test/nuscenes_info_0_108.json'

# # 读取 .pkl 文件
# with open(pkl_file_path, 'rb') as file:
#     data = pickle.load(file)

# # 递归函数，将 ndarray 转换为列表
# def convert_ndarray(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {k: convert_ndarray(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_ndarray(item) for item in obj]
#     else:
#         return obj

# rlt = []
# # 将 data['infos'][0] 转换为 JSON 格式
# for i in range(108):
#     data_json = convert_ndarray(data['infos'][i])
#     rlt.append(data_json)

# # 保存为 JSON 文件
# with open(json_file_path, 'w') as json_file:
#     json.dump(rlt, json_file, indent=4)

# print(f"Data has been saved to {json_file_path}")


#-----------------------------将一个.json文件保存为.pkl文件-------------------------
# # 定义文件路径
# json_file_path = '/home2/wzc/OccFormer/data_info_pre_test/nuscenes_info_01.json'
# pkl_file_path = '/home2/wzc/OccFormer/data_info_pre_test/nuscenes_info_01.pkl'

# # 读取 JSON 文件
# with open(json_file_path, 'r') as json_file:
#     data = json.load(json_file)

# # 递归函数，将特定的 list 还原为 ndarray
# def revert_ndarray(obj):
#     if isinstance(obj, list):
#         # 检查列表中的每个元素，如果是列表则转换为 ndarray
#         return np.array([revert_ndarray(i) for i in obj])
#     elif isinstance(obj, dict):
#         result = {}
#         for k, v in obj.items():
#             # 检查指定的键
#             if k in {'sweeps', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation'}:
#                 result[k] = v if isinstance(v, list) else revert_ndarray(v)
#             elif k == 'cams' and isinstance(v, dict):
#                 # 处理 cams 字典中的子元素
#                  # 处理 cams 字典中的每个子字典
#                 result[k] = {
#                     cam_key: 
#                     {
#                         sub_k: sub_v if isinstance(sub_v, list) and sub_k in 
#                         {'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation', 'ego2global_rotation'}
#                         else revert_ndarray(sub_v)
#                         for sub_k, sub_v in cam_val.items()
#                     } 
#                     for cam_key, cam_val in v.items()
#                 }
#             else:
#                 # 递归处理其他键
#                 result[k] = revert_ndarray(v)
#         return result
#     else:
#         return obj

# # 转换 JSON 数据中的特定列表为 ndarray
# data_reverted = [revert_ndarray(sample) for sample in data]

# # 保存为 .pkl 文件
# with open(pkl_file_path, 'wb') as pkl_file:
#     pickle.dump(data_reverted, pkl_file)

# print(f"Data has been saved to {pkl_file_path}")