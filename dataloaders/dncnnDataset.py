import torch

class DnCNNDataset(torch.utils.data.Dataset):
	def __init__(self, frames):
		# frames : list of tensor [C, H, W]
		self.frames = frames


	def __getitem__(self, idx):
		ret = {
			'noisy_input' : torch.unsqueeze(self.frames[idx], 0)
		}
		return ret


	def __len__(self):
		return len(self.frames)