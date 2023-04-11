import pickle, argparse, os
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm
from mayavi import mlab
from visualize_nusc_release import draw_nusc_occupancy

import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import pdb

def make_scene_videos(image_files, video_file):
	fps = 10
	size = [2930, 1640]
	video = cv2.VideoWriter(
		video_file, 
		cv2.VideoWriter_fourcc(*"MJPG"), 
		fps, 
		size,
	)
	
	image_files.sort()
	for image_file in image_files:
		img = cv2.imread(image_file)
		video.write(img)
	
	video.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('pred_dir', default=None)
	parser.add_argument('save_path', default=None)
	parser.add_argument('--scene-name', default=None)
	args = parser.parse_args()
	
	point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
	occ_size = [256, 256, 32]
	voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
	voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
	voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
	voxel_size = [voxel_x, voxel_y, voxel_z]
	
	# noqa
	constant_f = 0.0055
	scene_folders = os.listdir(args.pred_dir)
	save_path = args.save_path

	if args.scene_name is not None:
		assert args.scene_name in scene_folders
		print('processing scene {}'.format(args.scene_name))
	else:
		print('found {} scenes to process'.format(len(scene_folders)))

	for scene_index, scene_name in enumerate(scene_folders):
		if args.scene_name is not None:
			scene_name = args.scene_name

		scene_folder = os.path.join(args.pred_dir, scene_name)

		# scene folder assets
		scene_save_folder = os.path.join(save_path, '{}_assets'.format(scene_name))
		video_file = os.path.join(save_path, '{}_demo.avi'.format(scene_name))
		
		sample_img_files = []
		
		sample_files = os.listdir(scene_folder)
		sample_files.sort()

		for index, sample_file in tqdm(enumerate(sample_files), total=len(sample_files)):
			if not sample_file[-3:] == 'pkl':
				continue

			timestamp = sample_file.split('.')[0]
			sample_file = os.path.join(scene_folder, sample_file)

			sample_save_folder = os.path.join(scene_save_folder, '{}_assets'.format(timestamp))
			os.makedirs(sample_save_folder, exist_ok=True)

			sample_cat_file = os.path.join(scene_save_folder, '{}_cat_vis.png'.format(timestamp))
			sample_img_files.append(sample_cat_file)

			if os.path.exists(sample_cat_file):
				continue
			
			with open(sample_file, 'rb') as f:
				sample_data = pickle.load(f)
			
			pred_voxels = sample_data['pred_voxels']
			cam2lidar = sample_data['cam2lidar']
			img_canvas = sample_data['img_canvas']
			
			cam_positions = cam2lidar @ np.array([0., 0., 0., 1.])
			cam_positions = cam_positions[:, :3]
			focal_positions = cam2lidar @ np.array([0., 0., constant_f, 1.])
			focal_positions = focal_positions[:, :3]

			draw_nusc_occupancy(
				input_imgs=img_canvas,
				voxels=pred_voxels, 
				vox_origin=np.array(point_cloud_range[:3]),
				voxel_size=np.array(voxel_size),
				grid=np.array(occ_size),
				pred_lidarseg=None,
				target_lidarseg=None,
				save_folder=sample_save_folder,
				cat_save_file=sample_cat_file,
				cam_positions=cam_positions,
				focal_positions=focal_positions,
			)
		
		# create scene video
		make_scene_videos(sample_img_files, video_file)
		
		if args.scene_name is not None:
			break