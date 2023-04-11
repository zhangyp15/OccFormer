# Step-by-step installation instructions

OccFormer is developed based on the official BEVFormer codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate**

python 3.8 may not be supported.
```shell
conda create -n occformer python=3.7 -y
conda activate occformer
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
We select this pytorch version because mmdet3d 0.17.1 do not supports pytorch >= 1.11 and our cuda version is 11.3.

**c. Install mmcv, mmdet, and mmseg**
```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**c. Install mmdet3d 0.17.1**

Compared with the offical version, the mmdetection3d folder in this repo further includes operations like bev-pooling, voxel pooling. 

```shell
cd mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ..
```

**d. Install other dependencies, like timm, einops, torchmetrics, etc.**
```shell
pip install -r docs/requirements.txt
```