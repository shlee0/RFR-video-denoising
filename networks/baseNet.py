import abc
import torch


class BaseNet(torch.nn.Module, metaclass=abc.ABCMeta):
	def __init__(self, name):
		super(BaseNet, self).__init__()
		self.name = name