import os
import torch
from ignite.metrics import Average
from piq import psnr, ssim

from .baseModel import BaseModel

class Denoising(BaseModel):
	def __init__(self, denoiser, criterion=None, optimizer=None):
		super(Denoising, self).__init__('Denoising')

		self.denoiser = denoiser
		self.criterion = criterion
		self.optimizer = optimizer
		self.metrics = {
			'psnr' : Average(),
			'ssim' : Average()
		}

	@torch.no_grad()
	def evaluate(self, clean, denoised, device):
		clean = clean.to(device)
		denoised = denoised.to(device)
		denoised = torch.clamp(denoised, 0.0, 1.0)
		psnr_val = psnr(clean, denoised)
		ssim_val = ssim(clean, denoised)

		logs = {
			'psnr' : psnr_val,
			'ssim' : ssim_val
		}
		return logs


	def train_step(self, clean_target, input_dict, device):
		self.optimizer.zero_grad()
		denoised = self.forward(input_dict, device)
		loss = self.criterion(denoised, clean_target.to(device))
		loss.backward()
		self.optimizer.step()
		

	def forward(self, input_dict, device):
		return self.denoiser(input_dict, device)


	def save(self, dir_name='', file_name=None):
		self.denoiser.save(dir_name, file_name)


	def load(self, dir_name='', file_name=None):
		self.denoiser.load(dir_name, file_name)


	def get_log_path(self):
		return self.denoiser.name