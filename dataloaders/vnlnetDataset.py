"""
https://github.com/axeldavy/vnlnet
"""


import torch
import torchvision.transforms as transforms
import numpy as np

from video_patch_search import VideoPatchSearch

border_size = 41//2
ps = VideoPatchSearch(patch_search_width=41, patch_data_width=1,
						input_dtype=np.float32, past_frames=7,
						future_frames=7,
						search_width=41)


def rgb_to_gray(img):
	img = np.asarray(img, dtype=np.float32)
	if (len(img.shape) == 3 or img.shape[3] == 1):
		return img
	res = np.dot(img[...,:3], [0.57735, 0.57735, 0.57735])
	return np.asarray(res, dtype=np.float32)


def ps_func(noisy_frames):
	video = video_noised = np.asarray(noisy_frames)

	# Add a black border spatiotemporally
	video_noised_bigger = np.zeros([video.shape[0] + 7 + 7, video.shape[1] + 41-1, video.shape[2] + 41-1, video.shape[3]], dtype=np.float32)
	temporal_slice = slice(7, video_noised_bigger.shape[0]-7)

	video_noised_bigger[temporal_slice, border_size:-border_size, border_size:-border_size, :] = video_noised[:,:,:,:]

	# Replace black frames with future (or past frames)
	# Make it so different frames are seen during denoising
	for i in range(7):
		video_noised_bigger[i, border_size:-border_size, border_size:-border_size, :] = video_noised_bigger[7+7+1+i, border_size:-border_size, border_size:-border_size, :]
	for i in range(7):
		video_noised_bigger[-i-1, border_size:-border_size, border_size:-border_size, :] = video_noised_bigger[-(7+7+1+i)-1, border_size:-border_size, border_size:-border_size, :]

	video_search_gray = rgb_to_gray(video_noised_bigger)
	if len(video_search_gray.shape) == 3:
		video_search_gray = video_search_gray[:,:,:,np.newaxis]

	return video_noised_bigger, video_search_gray


def np_to_tensor(img):
	# in : 3d
	# out : 4d
	img = img.transpose(2, 0, 1)
	img = np.ascontiguousarray(img)
	img = np.expand_dims(img, 0)
	img = torch.Tensor(img)
	return img


class VNLnetDataset(torch.utils.data.Dataset):
	def __init__(self, frames):
		# frames : list of tensor [C, H, W]
		self.frames = []
		for i in range(len(frames)):
			frame = np.asarray(frames[i]).transpose((1,2,0))
			self.frames.append(frame)
		self.video_noised_bigger, self.video_search_gray = ps_func(self.frames)


	def __getitem__(self, idx):
		video_extract = self.video_noised_bigger[idx:(idx+7+1+7),:,:,:]
		video_search_extract_gray = self.video_search_gray[idx:(idx+7+1+7),:,:,:]
		nearest_neighbors_indices = ps.compute(video_search_extract_gray, 7)
		img_noised_patch_stack = ps.build_neighbors_array(video_extract, nearest_neighbors_indices[:-(2*border_size),:-(2*border_size),:])
		img_noised_patch_stack = np_to_tensor(img_noised_patch_stack)

		ret = {
			'noisy_input' : img_noised_patch_stack
		}
		return ret


	def __len__(self):
		return len(self.frames)