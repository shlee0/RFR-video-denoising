import abc, os
import torch


class BaseModel(torch.nn.Module, metaclass=abc.ABCMeta):
	def __init__(self, name):
		super(BaseModel, self).__init__()
		self.name = name
		
		
	def reset_metrics(self):
		for key in self.metrics:
			self.metrics[key].reset()


	def update_metrics(self, logs):
		for key in self.metrics:
			self.metrics[key].update(logs[key])
			

	def result(self):
		rets = {}
		for key in self.metrics:
			rets[key] = self.metrics[key].compute().cpu().numpy()
		return rets