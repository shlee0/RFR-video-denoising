"""
https://github.com/m-tassano/fastdvdnet
"""


import torch

class FastDVDnetDataset(torch.utils.data.Dataset):
	def __init__(self, frames, sigma):
		# frames : list of tensor [C, H, W]
		self.frames = frames
		self.sigma = sigma


	def get_neighbor_frames(self, idx):
		ret = []
		frame_len = len(self)
		for d in range(-2, 3):
			i = idx + d
			if i < 0:
				i = -i
			if i >= frame_len:
				last_idx = frame_len - 1
				i = last_idx - (i - last_idx)
			ret.append(self.frames[i])
		return ret


	def __getitem__(self, idx):
		# input noisy frame
		noisy_stack = self.get_neighbor_frames(idx)
		noisy_stack = torch.cat(noisy_stack, 0) # channel concat
		noisy_stack = torch.unsqueeze(noisy_stack, 0)	

		# noise map
		_, h, w = self.frames[0].shape
		sigma_noise = torch.ones(1, 1, h, w) * (self.sigma / 255)

		ret = {
			'noisy_input' : noisy_stack,
			'noise_map' : sigma_noise
		}
		return ret


	def __len__(self):
		return len(self.frames)