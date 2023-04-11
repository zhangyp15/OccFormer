## SemanticKITTI

To prepare for SemanticKITTI dataset, please download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Put all `.zip` files under `OccFormer/data/SemanticKITTI` and unzip these files. Then you should get the following dataset structure:
```
OccFormer
├── data/
│   ├── SemanticKITTI/
│   │   ├── data_velodyne/
│   │   │   │   ├── velodyne/
│   │   ├── dataset/
│   │   │   ├── sequences
│   │   │   │   ├── 00
│   │   │   │   │   ├── calib.txt
│   │   │   │   │   ├── image_2/
│   │   │   │   │   ├── image_3/
│   │   │   │   │   ├── voxels/
│   │   │   │   ├── 01
│   │   │   │   ├── 02
│   │   │   │   ├── ...
│   │   │   │   ├── 21
```

Preprocess the annotations for semantic scene completion:
```bash
python projects/mmdet3d_plugin/tools/kitti_process/semantic_kitti_preprocess.py --kitti_root data/SemanticKITTI --kitti_preprocess_root data/SemanticKITTI --data_info_path projects/mmdet3d_plugin/tools/kitti_process/semantic-kitti.yaml
```

## NuScenes
Please download **nuScenes full dataset v1.0**, **CAN bus expansion**, and **nuScenes-lidarseg** from the [official website](https://www.nuscenes.org/download). The dataset folder should be organized as follows:
```
OccFormer
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── lidarseg
|   |   │   ├──v1.0-trainval/
|   |   │   ├──v1.0-mini/
|   |   │   ├──v1.0-test/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
|   |   ├── nuscenes_infos_temporal_test.pkl
```

To generate the above data infos, directly download [infos](https://github.com/zhangyp15/OccFormer/releases/tag/data_infos) or prepare yourself by running:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data --extra-tag nuscenes --version v1.0 --canbus ./data
```

