import re
import cv2
from datetime import datetime, timedelta

# 视频文件路径
video_path = "/home2/wzc/OccFormer/fsc_data/video/1731393926_3.mp4"
# 图片保存路径
output_folder = "/home2/wzc/OccFormer/fsc_data/data/images_3/CAM_FRONT"

# 从文件路径中提取Unix时间戳
match = re.search(r"/(\d+)_\d+\.mp4$", video_path)
if match:
    unix_timestamp = int(match.group(1))
    start_time = datetime.utcfromtimestamp(unix_timestamp)
    start_timestamp = start_time.strftime("%Y_%m_%d_%H%M%S") + f"_{unix_timestamp}000000"
    print("视频起始时间戳（微秒）:", start_timestamp)
else:
    print("未能从文件路径中提取出Unix时间戳")
    start_timestamp = None

# 使用正则表达式从 output_folder 提取相机名称
cam_match = re.search(r"/([^/]+)$", output_folder)
if cam_match:
    cam_name = cam_match.group(1)
    print("相机名称:", cam_name)
else:
    cam_name = "UNKNOWN"
    print("未能从文件路径中提取出相机名称")

# 帧率（每0.5秒提取一帧）
frame_rate = 2  # Hz
time_increment = timedelta(seconds=0.5)  # 每帧时间间隔为 0.5 秒

if start_timestamp:
    # 将起始时间戳转换为datetime对象
    start_time = datetime.strptime(start_timestamp[:17], "%Y_%m_%d_%H%M%S")
    current_time = start_time
    current_unix_time = int(f"{unix_timestamp}000000")  # 初始 Unix 时间戳（微秒级）

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)  # 每隔多少帧提取一帧

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检查是否到了要保存的帧
        if frame_count % frame_interval == 0:
            # 格式化时间戳并命名文件，包含相机名称
            timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S") + f"_{cam_name}_{current_unix_time}"
            output_path = f"{output_folder}/{timestamp_str}.jpg"
            cv2.imwrite(output_path, frame)

            # 更新当前时间和 Unix 时间戳
            current_time += time_increment
            current_unix_time += 500000  # 每帧递增 500,000 微秒 (0.5 秒)

        frame_count += 1

    # 释放视频资源
    cap.release()
