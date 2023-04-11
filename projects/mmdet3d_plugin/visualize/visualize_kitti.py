import pickle, argparse, os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import mayavi

from tqdm import tqdm
from PIL import Image
from mayavi import mlab
mlab.options.offscreen = True

import pdb

''' class names:
'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
'pole', 'traffic-sign'
'''
colors = np.array(
	[
		[100, 150, 245, 255],
		[100, 230, 245, 255],
		[30, 60, 150, 255],
		[80, 30, 180, 255],
		[100, 80, 250, 255],
		[255, 30, 30, 255],
		[255, 40, 200, 255],
		[150, 30, 90, 255],
		[255, 0, 255, 255],
		[255, 150, 255, 255],
		[75, 0, 75, 255],
		[175, 0, 75, 255],
		[255, 200, 0, 255],
		[255, 120, 50, 255],
		[0, 175, 0, 255],
		[135, 60, 0, 255],
		[150, 240, 80, 255],
		[255, 240, 150, 255],
		[255, 0, 0, 255],
	]).astype(np.uint8)


def get_grid_coords(dims, resolution):
	"""
	:param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
	:return coords_grid: is the center coords of voxels in the grid
	"""

	g_xx = np.arange(0, dims[0] + 1)
	g_yy = np.arange(0, dims[1] + 1)
	sensor_pose = 10
	g_zz = np.arange(0, dims[2] + 1)

	# Obtaining the grid with coords...
	xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
	coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
	coords_grid = coords_grid.astype(np.float32)

	coords_grid = (coords_grid * resolution) + resolution / 2

	temp = np.copy(coords_grid)
	temp[:, 0] = coords_grid[:, 1]
	temp[:, 1] = coords_grid[:, 0]
	coords_grid = np.copy(temp)

	return coords_grid

def draw(
	voxels,
	T_velo_2_cam,
	vox_origin,
	fov_mask,
	img_size,
	f,
	voxel_size=0.2,
	d=7,  # 7m - determine the size of the mesh representing the camera
	save_name=None,
	video_view=True,
):
	# Compute the voxels coordinates
	grid_coords = get_grid_coords(
		[voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
	)

	# Attach the predicted class to every voxel
	grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

	# Get the voxels inside FOV
	fov_grid_coords = grid_coords[fov_mask, :]

	# Get the voxels outside FOV
	outfov_grid_coords = grid_coords[~fov_mask, :]

	# Remove empty and unknown voxels
	fov_voxels = fov_grid_coords[
		(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
	]
	outfov_voxels = outfov_grid_coords[
		(outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
	]

	figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

	''' Draw the camera '''
	if T_velo_2_cam is not None:
		x = d * img_size[0] / (2 * f)
		y = d * img_size[1] / (2 * f)
		tri_points = np.array(
			[
				[0, 0, 0],
				[x, y, d],
				[-x, y, d],
				[-x, -y, d],
				[x, -y, d],
			]
		)
		tri_points = np.hstack([tri_points, np.ones((5, 1))])
		tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
		x = tri_points[:, 0] - vox_origin[0]
		y = tri_points[:, 1] - vox_origin[1]
		z = tri_points[:, 2] - vox_origin[2]
		triangles = [
			(0, 1, 2),
			(0, 1, 4),
			(0, 3, 4),
			(0, 2, 3),
		]
		
		mlab.triangular_mesh(
			x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
		)

	# Draw occupied inside FOV voxels
	plt_plot_fov = mlab.points3d(
		fov_voxels[:, 0],
		fov_voxels[:, 1],
		fov_voxels[:, 2],
		fov_voxels[:, 3],
		colormap="viridis",
		scale_factor=voxel_size - 0.05 * voxel_size,
		mode="cube",
		opacity=1.0,
		vmin=1,
		vmax=19,
	)

	infov_colors = colors
	plt_plot_fov.glyph.scale_mode = "scale_by_vector"
	plt_plot_fov.module_manager.scalar_lut_manager.lut.table = infov_colors

	# Draw occupied outside FOV voxels
	if outfov_voxels.shape[0] > 0:
		plt_plot_outfov = mlab.points3d(
			outfov_voxels[:, 0],
			outfov_voxels[:, 1],
			outfov_voxels[:, 2],
			outfov_voxels[:, 3],
			colormap="viridis",
			scale_factor=voxel_size - 0.05 * voxel_size,
			mode="cube",
			opacity=1.0,
			vmin=1,
			vmax=19,
		)

		outfov_colors = colors
		outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
		plt_plot_outfov.glyph.scale_mode = "scale_by_vector"
		plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

	scene = figure.scene
	if video_view:
		scene.camera.position = [-96.17897208968986, 24.447806140326282, 71.4786454057558]
		scene.camera.focal_point = [25.59999984735623, 25.59999984735623, 2.1999999904073775]
		scene.camera.view_angle = 23.999999999999993
		scene.camera.view_up = [0.4945027163799531, -0.004902474180369383, 0.8691622571417599]
		scene.camera.clipping_range = [91.71346136213631, 201.25874270827438]
	else:
		scene.camera.position = [-50.907238103376244, -51.31911151935225, 104.75510851395386]
		scene.camera.focal_point = [23.005321731256945, 23.263153155247394, 0.7241134057028675]
		scene.camera.view_angle = 19.199999999999996
		scene.camera.view_up = [0.5286546999662366, 0.465851763212298, 0.7095818084728509]
		scene.camera.clipping_range = [92.25158502285397, 220.40602072417923]
	
	scene.camera.compute_view_plane_normal()
	scene.render()

	save_file = save_name + '.png'
	mlab.savefig(save_file)
	print('saving to {}'.format(save_file))

	return save_file

def make_video(result_folder, video_file):
	fps = 12
	size = [700, 650]
	font_size = 1
	thickness = 2
	font = cv2.FONT_HERSHEY_COMPLEX

	video = cv2.VideoWriter(
		video_file, 
		cv2.VideoWriter_fourcc(*"MJPG"), 
		fps, 
		size,
	)

	for index, image_file in enumerate(tqdm(os.listdir(result_folder))):
		if not image_file[-3:] == 'png':
			continue

		img = os.path.join(result_folder, image_file)
		img = cv2.imread(img)

		# add method name
		cv2.putText(img, 'Input image', (250, 240), font, font_size, (0, 0, 0), thickness)
		cv2.putText(img, 'OccFormer(ours)', (230, 640), font, font_size, (0, 0, 0), thickness)
	
		# in case that some images are broken
		if img is None:
			continue

		video.write(img)

	video.release()
	cv2.destroyAllWindows()

def main_draw():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('pred_dir', default=None)
	parser.add_argument('save_path', default=None)
	args = parser.parse_args()
	
	# fixed parameters
	vox_origin = np.array([0, -25.6, -2])
	occformer_path = args.pred_dir
 
	# prepare output paths
	save_path = args.save_path
	assets_path = os.path.join(save_path, 'assets')
	os.makedirs(assets_path, exist_ok=True)

	resize_img_size = [700, 200]
	spacing = 0

	samples = [x for x in os.listdir(occformer_path) if x[:2] == '00']
	samples.sort()
	print('found total {} samples'.format(len(samples)))

	for sample in samples:
		sample_id = sample.split('.')[0]
		occformer_sample = os.path.join(occformer_path, sample)
		occ_file = os.path.join(assets_path, '{}_occ'.format(sample_id))
		result_file = os.path.join(save_path, '{}_vis.png'.format(sample_id))

		if os.path.exists(result_file):
			continue

		# load occformer outputs
		with open(occformer_sample, "rb") as handle:
			occformer_data = pickle.load(handle)

		fov_mask = np.ones(occformer_data['output_voxel'].shape[0]).astype(bool)
		occformer_pred = occformer_data['output_voxel'].reshape(256, 256, 32)
		input_image = occformer_data['raw_img'][..., [2, 1, 0]] # BGR to RGB

		# draw occformer predictions
		occformer_img = draw(occformer_pred, None, vox_origin, fov_mask, 
			img_size=(1220, 370), f=707.0912, voxel_size=0.2, d=7, save_name=occ_file, video_view=True)

		# cat the input image and the occupancy predictions
		occformer_img = Image.open(occformer_img)
		occ_raw_w, occ_raw_h = occformer_img.size
		top_crop = 300
		bottom_crop = 200
		occformer_img = occformer_img.crop([0, top_crop, occ_raw_w, occ_raw_h - bottom_crop])

		occ_raw_w, occ_raw_h = occformer_img.size
		occ_h = int(resize_img_size[0] / occ_raw_w * occ_raw_h)
		occ_img_size = [resize_img_size[0], occ_h]
		occformer_img = occformer_img.resize(occ_img_size, Image.BILINEAR)
		input_image = Image.fromarray(input_image).resize(resize_img_size, Image.BILINEAR)

		# 竖向排布
		output_w = resize_img_size[0]
		output_h = resize_img_size[1] + spacing + occ_img_size[1]
		result = Image.new(input_image.mode, (output_w, output_h), (255, 255, 255))
		result.paste(input_image, box=(0, 0))
		result.paste(occformer_img, box=(0, resize_img_size[1] + spacing))
		result.save(result_file)

		print('finish processing {}'.format(sample_id))
	
	video_file = os.path.join(save_path, 'kitti_demo.avi')
	make_video(save_path, video_file)

if __name__ == "__main__":
	main_draw()

