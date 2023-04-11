# Getting started with OccFormer

For most of our experiments, we train the model with 8 RTX 3090 GPUs with 24G memory. Unfortunately, we already use the pytorch checkpoint (for all models) and fp16 (only for R101-DCN experiment on nuScenes) to reduce the memory. Therefore, you may need similar hardwares to reproduce the training results.

Before start training, download the corresponding pretrained backbones from the [release page](https://github.com/zhangyp15/OccFormer/releases/tag/assets) and put them under the folder `ckpts/`. The weights include [EfficientNetB7](https://github.com/zhangyp15/OccFormer/releases/download/assets/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth) for KITTI and [R50](https://github.com/zhangyp15/OccFormer/releases/download/assets/resnet50-0676ba61.pth) & [R101-DCN](https://github.com/zhangyp15/OccFormer/releases/download/assets/r101_dcn_fcos3d_pretrain.pth) for nuScenes.

## Training
```bash
bash tools/dist_train.sh $CONFIG 8
```
During the training process, the model is evaluated on the validation set after every epoch. The checkpoint with best performance will be saved. The output logs and checkpoints will be available at work_dirs/$CONFIG.

## Evaluation
Evaluate with 1 GPU:
```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval=bbox
```
The single-GPU inference will print the current performance after each iteration, which can serve as a quick indicator.

Evaluate with 8 GPUs:
```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8
```

