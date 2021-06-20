import os, argparse

# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--net', type=str, choices=['vnlnet', 'fastdvdnet', 'dncnn'], required=True)
parser.add_argument('--video_dir', type=str, default='./testsets/derf_HD/factory_1080p30')
parser.add_argument('--sigma', type=int, choices=[15, 25, 40], default=25)
parser.add_argument('--online', action='store_true')
parser.add_argument('--offline', action='store_true')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
from PIL import Image
import collections

from models.denoising import Denoising

from utils import get_all_img_paths, save_png, make_dir



# gt root
VIDEO_ROOT = args.video_dir
SIGMA = args.sigma
FINETUNE_LR = 1e-5
PRE_TRAINED_CKPT_PATH = './results/ckpts/pre-trained'


# denoiser
if args.net == 'vnlnet':
	from networks.vnlnet import ModifiedDnCNN
	from dataloaders.vnlnetDataset import VNLnetDataset

	denoiser = ModifiedDnCNN()

	def get_dataloader(frames):
		return VNLnetDataset(frames)

	def load_pre_trained():
		denoiser.load_pre_trained(PRE_TRAINED_CKPT_PATH, SIGMA)


elif args.net == 'fastdvdnet':
	from networks.fastdvdnet import FastDVDnet
	from dataloaders.fastdvdnetDataset import FastDVDnetDataset

	denoiser = FastDVDnet()

	def get_dataloader(frames):
		return FastDVDnetDataset(frames, SIGMA)

	def load_pre_trained():
		denoiser.load_pre_trained(PRE_TRAINED_CKPT_PATH)


elif args.net == 'dncnn': # grayscale image
	from networks.dncnn import DnCNN
	from dataloaders.dncnnDataset import DnCNNDataset

	denoiser = DnCNN(channels=1)

	def get_dataloader(frames):
		return DnCNNDataset(frames)

	def load_pre_trained():
		denoiser.load_pre_trained(PRE_TRAINED_CKPT_PATH, sigma=25)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
denoiser = denoiser.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser.parameters(), FINETUNE_LR)
model = Denoising(denoiser, criterion, optimizer)




def gaussian_noisy(x, std):
	y = x + torch.distributions.normal.Normal(0, std / 255).sample(x.shape)
	y = y.clamp(0,1)
	return y




@torch.no_grad()
def evaluation(gt_frames, noisy_dataset, img_path=None):
	ret = {
		'psnr' : [],
		'ssim' : []
	}
	
	for i in range(len(gt_frames)):
		# gt
		img_gt = torch.unsqueeze(gt_frames[i], 0)

		# noisy
		input_dict = noisy_dataset[i]

		# denoised
		denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# metrics (psnr, ssim)
		logs = model.evaluate(img_gt, denoised, device)
		for key in logs:
			ret[key].append(logs[key].cpu().numpy())

		if img_path is not None:
			# save denoised
			denoised = denoised[0].cpu()
			save_png(denoised, os.path.join(img_path, '{:03}_{:0.3f}_{:0.5f}'.format(i, logs['psnr'], logs['ssim'])))
		
	return ret


def rfr_online(gt_frames, noisy_dataset, img_path):
	optimizer.state = collections.defaultdict(dict) # opt reinit
	ret = {
		'psnr' : [],
		'ssim' : []
	}

	frame_len = len(gt_frames)

	for i in range(frame_len):
		# gt
		img_gt = torch.unsqueeze(gt_frames[i], 0)

		# noisy
		input_dict = noisy_dataset[i]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# metrics (psnr, ssim)
		logs = model.evaluate(img_gt, denoised, device)
		for key in logs:
			ret[key].append(logs[key].cpu().numpy())

		# save denoised
		denoised = denoised[0].cpu()
		save_png(denoised, os.path.join(img_path, '{:03}_{:0.3f}_{:0.5f}'.format(i, logs['psnr'], logs['ssim'])))

		# fine-tune with pseudo training pair
		pseudo_clean = denoised.unsqueeze(0)
		pseudo_noisy = gaussian_noisy(pseudo_clean.cpu(), SIGMA)
		input_dict['noisy_input'] = pseudo_noisy

		model.train_step(pseudo_clean, input_dict, device)
		
	return ret


def rfr_online_multiInputFrames(gt_frames, noisy_dataset, img_path):
	optimizer.state = collections.defaultdict(dict) # opt reinit
	ret = {
		'psnr' : [],
		'ssim' : []
	}

	frame_len = len(gt_frames)
	psuedo_cleans = []
	input_dicts = []

	for i in range(frame_len):
		# noisy
		input_dict = noisy_dataset[i]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		psuedo_cleans.append(denoised[0].cpu())
		input_dicts.append(input_dict)
		
	# for fast fine-tuning
	psuedo_clean_dataset = get_dataloader(psuedo_cleans)


	for i in range(frame_len):
		# gt
		img_gt = torch.unsqueeze(gt_frames[i], 0)

		# noisy
		input_dict = input_dicts[i]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# metrics (psnr, ssim)
		logs = model.evaluate(img_gt, denoised, device)
		for key in logs:
			ret[key].append(logs[key].cpu().numpy())

		# save denoised
		denoised = denoised[0].cpu()
		save_png(denoised, os.path.join(img_path, '{:03}_{:0.3f}_{:0.5f}'.format(i, logs['psnr'], logs['ssim'])))

		# fine-tune with pseudo training pair
		pseudo_clean = torch.unsqueeze(psuedo_cleans[i], 0)
		pseudo_noisy = gaussian_noisy(psuedo_clean_dataset[i]['noisy_input'], SIGMA)
		input_dict['noisy_input'] = pseudo_noisy

		model.train_step(pseudo_clean, input_dict, device)

	return ret
	

def rfr_offline(gt_frames, noisy_dataset, ckpt_path, n_step):
	optimizer.state = collections.defaultdict(dict) # opt reinit

	frame_len = len(gt_frames)
	psuedo_cleans = []
	input_dicts = []

	for i in range(frame_len):
		# noisy
		input_dict = noisy_dataset[i]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		psuedo_cleans.append(denoised[0].cpu())
		input_dicts.append(input_dict)
		
	psuedo_clean_dataset = get_dataloader(psuedo_cleans)


	for step in range(n_step):
		psuedo_cleans_ft = []
		for i in range(frame_len):
			# noisy
			input_dict = input_dicts[i]

			# denoised
			with torch.no_grad():
				denoised = torch.clamp(model(input_dict, device), 0., 1.)

			psuedo_cleans_ft.append(denoised[0].cpu())
			

		psuedo_clean_ft_dataset = get_dataloader(psuedo_cleans_ft)

		# random permute frames
		idxes = np.arange(frame_len)
		np.random.shuffle(idxes)
		for i in idxes:
			# pseudo noisy
			pseudo_clean = psuedo_cleans[i]
			input_dict = psuedo_clean_dataset[i]
			input_dict['noisy_input'] = gaussian_noisy(input_dict['noisy_input'], SIGMA)

			pseudo_clean_ft = psuedo_cleans_ft[i]
			input_dict_ft = psuedo_clean_ft_dataset[i]
			input_dict_ft['noisy_input'] = gaussian_noisy(input_dict_ft['noisy_input'], SIGMA)


			# concat
			for key in input_dict:
				input_dict[key] = torch.cat((input_dict[key], input_dict_ft[key]), 0)

			pseudo_clean = torch.stack((pseudo_clean, pseudo_clean_ft), 0)

			# finetune
			model.train_step(pseudo_clean, input_dict, device)

		checkpoint_path = os.path.join(ckpt_path, '{:02}'.format(step + 1) + '.pt')
		torch.save(denoiser.state_dict(), checkpoint_path)





if __name__ == '__main__':
	# eval mode for batch-norm layers
	denoiser.eval()

	
	splits = VIDEO_ROOT.split('/')
	dataset_name, video_name = splits[-2], splits[-1]
	print (dataset_name, video_name)

	result_path = os.path.join('results', model.get_log_path(), dataset_name, video_name)
	img_path = os.path.join(result_path, 'imgs')
	ckpt_path = os.path.join(result_path, 'ckpts')
	make_dir(ckpt_path)

	# load frames
	gt_frames = []
	gt_frame_paths = get_all_img_paths(VIDEO_ROOT)
	for i in range(len(gt_frame_paths)):
		gt_frames.append(transforms.ToTensor()(Image.open(gt_frame_paths[i])))


	noisy_frames = []
	for i in range(len(gt_frames)):
		noisy_frames.append(gaussian_noisy(gt_frames[i], SIGMA))

	noisy_dataset = get_dataloader(noisy_frames)


	if 1:
		print (video_name, 'evaluation')

		load_pre_trained()

		ret = evaluation(gt_frames, noisy_dataset, img_path)
		print ('evaluation results')
		for i in range(len(ret['psnr'])):
			print (i, ret['psnr'][i], ret['ssim'][i])



	if args.online:
		print (video_name, 'rfr_online')

		load_pre_trained()
		
		if args.net == 'dncnn':
			ret = rfr_online(gt_frames, noisy_dataset, img_path + '_online')
		else:
			ret = rfr_online_multiInputFrames(gt_frames, noisy_dataset, img_path + '_online')
		
		print ('rfr_online results')
		for i in range(len(ret['psnr'])):
			print (i, ret['psnr'][i], ret['ssim'][i])



	if args.offline:
		print (video_name, 'rfr_offline')

		load_pre_trained()

		n_step = 10
		rfr_offline(gt_frames, noisy_dataset, ckpt_path, n_step)

		for step in range(n_step):
			checkpoint_path = os.path.join(ckpt_path, '{:02}'.format(step + 1) + '.pt')
			state = torch.load(checkpoint_path)
			denoiser.load_state_dict(state)

			ret = evaluation(gt_frames, noisy_dataset)
			print ('rfr_offline {:02} results'.format(step + 1))
			for i in range(len(ret['psnr'])):
				print (i, ret['psnr'][i], ret['ssim'][i])