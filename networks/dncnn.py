"""
https://github.com/SaoYan/DnCNN-PyTorch
"""

import os
import torch
import torch.nn as nn
from .baseNet import BaseNet

class DnCNN(BaseNet):
	def __init__(self, channels, num_of_layers=17):
		super(DnCNN, self).__init__('DnCNN')
		kernel_size = 3
		padding = 1
		features = 64
		layers = []
		layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(num_of_layers - 2):
			layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
			layers.append(nn.BatchNorm2d(features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
		self.dncnn = nn.Sequential(*layers)

	def forward(self, input_dict, device):
		x = input_dict['noisy_input'].to(device)
		out = self.dncnn(x)
		return x - out

	def load_pre_trained(self, dir_path, sigma):
		checkpoint_path = os.path.join(dir_path, 'dncnn_{}.pth'.format(sigma))
		_state = torch.load(checkpoint_path)
		state = {}
		for key in _state:
			state[key.replace('module.', '')] = _state[key]
		self.load_state_dict(state)