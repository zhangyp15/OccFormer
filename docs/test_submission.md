We provide the guidance for preparing the test submissions for Semantic Scene Completion on SemanticKITTI and for LiDAR Segmentation on nuScenes.

## SemanticKITTI
Following the practice from [MonoScene](https://github.com/astra-vision/MonoScene/issues/54), we train OccFormer only on the train sequences and select the checkpoint with best validation performance for test submission. This is how we get the benchmark results in the paper.

To prepare the test submission, we first inference OccFormer on test sequences:
```bash
bash tools/dist_test.sh projects/configs/occformer_kitti/occformer_kitti_submit.py $YOUR_CKPT 8 --test-save $YOUR_SAVE_PATH
```

Compress the predictions:
```bash
cd $YOUR_SAVE_PATH && zip -r $ZIP_FILE sequences
```

Then, validate the `$ZIP_FILE` with [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api):
```bash
python projects/mmdet3d_plugin/tools/validate_semkitti_submission.py $ZIP_FILE --dataset data/SemanticKITTI/dataset
```

If everything is ready after the validation, submit the `$ZIP_FILE` to the [competition site](https://codalab.lisn.upsaclay.fr/competitions/7170#participate) and check the results.

## nuScenes

First, we train OccFormer with train+val scenes:
```bash
bash tools/dist_train.sh projects/configs/occformer_nusc/occformer_nusc_r101_896x1600_trainval.py 8
```

Second, we inference OccFormer on test scenes with the final checkpoint:
```bash
bash tools/dist_test.sh projects/configs/occformer_nusc/occformer_nusc_r101_896x1600_trainval.py work_dirs/occformer_nusc_r101_896x1600_trainval/latest.pth 8 --test-save $YOUR_SAVE_PATH
```

Then, validate your predictions with the script from [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit):
```bash
python projects/mmdet3d_plugin/tools/validate_lidarseg_submission.py --result-path $YOUR_SAVE_PATH --dataroot data/nuscenes --zip-out .
```
The validation script will check the necessary information and also compress your results into a .zip file for submission. 

Finally, submit `$RESULT_FILE` to Eval.ai. We recommend to use the evalai-cli package for uploading. Please check the [submit page](https://eval.ai/web/challenges/challenge-page/720/submission) for more details.