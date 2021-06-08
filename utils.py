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

def save_png(img, path):
	if type(img) != Image.Image:
		img = transforms.ToPILImage()(img)
	path = os.path.abspath(path)
	dir_path = os.path.dirname(path)
	make_dir(dir_path)
	img.save(path + '.png')