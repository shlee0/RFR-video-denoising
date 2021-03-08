import os
from PIL import Image

import torchvision.transforms as transforms

def make_dir(dir_path):
	if not(os.path.isdir(dir_path)):
		os.makedirs(dir_path)
	
def get_all_img_paths(root_path):
	paths = []
	for (dirpath, dirnames, filenames) in os.walk(root_path):
		filenames = [f for f in filenames if not f[0] == '.']
		dirnames[:] = [d for d in dirnames if not d[0] == '.']

		for filename in filenames:
			if (filename.lower().endswith(tuple(['.bmp', '.jpg', '.png']))):
				path = os.path.join(dirpath, filename)
				paths.append(path)
	return sorted(paths)

def get_frame_paths(paths):
	ret = []
	tmp_s = set()

	for p in paths:
		frame_dir_path = os.path.dirname(p)
		file_name = os.path.basename(p)

		if not frame_dir_path in tmp_s:
			tmp_s.add(frame_dir_path)
			ret.append([frame_dir_path, []])

		ret[-1][1].append(file_name)
	return ret

def get_folder_img_name(img_path):
	img_path = os.path.abspath(img_path)
	img_path_split = img_path.split('/')
	folder_name = img_path_split[-2]
	img_name = img_path_split[-1]
	img_name = os.path.splitext(img_name)[0]
	return folder_name, img_name

def save_png(img, path):
	if type(img) != Image.Image:
		img = transforms.ToPILImage()(img)
	path = os.path.abspath(path)
	dir_path = os.path.dirname(path)
	make_dir(dir_path)
	img.save(path + '.png')