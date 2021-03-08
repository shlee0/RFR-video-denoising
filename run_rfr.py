import os
from config import parser, TEST_DS_DICT

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
from PIL import Image
import functools
import datetime
import collections
import pickle

from models.denoising import Denoising

from utils import get_all_img_paths, save_png, make_dir, get_frame_paths, get_folder_img_name


# args parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='-1')
parser.add_argument('--net', type=str, choices=['vnlnet', 'fastdvdnet', 'dncnn'], required=True)
parser.add_argument('--test', type=str, choices=[
	'DERF_HD_15', 'DERF_HD_25', 'DERF_HD_40',
	'DERF_HD_GRAY_15', 'DERF_HD_GRAY_25', 'DERF_HD_GRAY_40',
	'DAVIS_15', 'DAVIS_25', 'DAVIS_40'
], required=True)
parser.add_argument('--offline', action='store_true')
parser.add_argument('--iter', type=int, default='1')
parser.add_argument('--rebuttal', action='store_true')


# gt root




DEBUG = 0

# hyperparameter
SIGMA = int(args.test.split('_')[-1])

FINETUNE_LR = 1e-5

EVAL_GT_ROOT = TEST_DS_DICT[args.test]['clean']
EVAL_NOISY_ROOT = TEST_DS_DICT[args.test]['noisy']


# test dataset
eval_gt_paths = get_all_img_paths(EVAL_GT_ROOT)
eval_noisy_paths = get_all_img_paths(EVAL_NOISY_ROOT)



eval_gt_frame_paths = get_frame_paths(eval_gt_paths)
eval_noisy_frame_paths = get_frame_paths(eval_noisy_paths)


if DEBUG:
	for i in range(len(eval_gt_frame_paths)):
		eval_gt_frame_paths[i][1] = eval_gt_frame_paths[i][1][:30]
		eval_noisy_frame_paths[i][1] = eval_noisy_frame_paths[i][1][:30]


# model
if args.net == 'vnlnet':
	from networks.vnlnet import ModifiedDnCNN
	from datasets.vnlnetDataset import VNLnetDataset

	denoiser = ModifiedDnCNN()

	def get_dataset(frames):
		return VNLnetDataset(frames)

	def load_denoiser():
		denoiser.load(SIGMA)


elif args.net == 'fastdvdnet':
	from networks.fastdvdnet import FastDVDnet
	from datasets.fastdvdnetDataset import FastDVDnetDataset

	denoiser = FastDVDnet()

	def get_dataset(frames):
		return FastDVDnetDataset(frames, SIGMA)

	def load_denoiser():
		denoiser.load()


elif args.net == 'dncnn': # grayscale image
	from networks.dncnn import DnCNN
	from datasets.dncnnDataset import DnCNNDataset

	denoiser = DnCNN(channels=1)

	def get_dataset(frames):
		return DnCNNDataset(frames)

	def load_denoiser():
		denoiser.load(sigma=25)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
denoiser = denoiser.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser.parameters(), FINETUNE_LR)
model = Denoising(denoiser, criterion, optimizer)



def finetuning_mode():
	for m in denoiser.modules():
		if isinstance(m, nn.BatchNorm2d):
			m.eval()
		else:
			m.train()

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
	
	for i in range(len(noisy_frames)):
		# gt
		img_gt = torch.unsqueeze(gt_frames[i], 0)

		# noisy
		input_dict = noisy_dataset[i]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# metrics
		logs = model.evaluate(img_gt, denoised, device)
		for key in logs:
			ret[key].append(logs[key].cpu().numpy())

		if img_path is not None and i < 100:
			denoised = denoised[0].cpu()
			denoised = transforms.ToPILImage()(denoised)
			# save_png(denoised, os.path.join(img_path, '{:03}_{:0.3f}_{:0.5f}'.format(i, logs['psnr'], logs['ssim'])))
		
	return ret


def rfr_online(gt_frames, noisy_dataset, img_path):
	optimizer.state = collections.defaultdict(dict) # opt reinit
	ret = {
		'psnr' : [],
		'ssim' : []
	}

	frame_len = len(gt_frames)
	psuedo_clean_frames = []
	input_dicts = []

	for fi in range(frame_len):
		# noisy
		input_dict = noisy_dataset[fi]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		psuedo_clean_frames.append(denoised[0].cpu())
		input_dicts.append(input_dict)
		

	# rfr
	psuedo_clean_ds = get_dataset(psuedo_clean_frames)

	for fi in range(frame_len):
		# gt
		img_gt = torch.unsqueeze(gt_frames[fi], 0)

		# noisy
		input_dict = input_dicts[fi]

		# denoised frame
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# metrics
		logs = model.evaluate(img_gt, denoised, device)
		for key in logs:
			ret[key].append(logs[key].cpu().numpy())

		if img_path is not None and fi < 100:
			denoised = denoised[0].cpu()
			denoised = transforms.ToPILImage()(denoised)
			save_png(denoised, os.path.join(img_path, '{:03}_{:0.3f}_{:0.5f}'.format(fi, logs['psnr'], logs['ssim'])))


		for _ in range(args.iter): # comparison to F2F
			# pseudo noisy
			input_dict = psuedo_clean_ds[fi]
			input_dict['noisy_input'] = gaussian_noisy(input_dict['noisy_input'], SIGMA)
			pseudo_clean_frame = torch.unsqueeze(psuedo_clean_frames[fi], 0)

			# finetune
			# import time
			# ss = time.time()
			model.train_step(pseudo_clean_frame, input_dict, device)
			# print (time.time() - ss)
			# input ('c')

	return ret


def rfr_online_dncnn(gt_frames, noisy_dataset, img_path):
	optimizer.state = collections.defaultdict(dict) # opt reinit
	ret = {
		'psnr' : [],
		'ssim' : []
	}

	frame_len = len(gt_frames)

	for fi in range(frame_len):
		# gt
		img_gt = torch.unsqueeze(gt_frames[fi], 0)

		# noisy
		input_dict = noisy_dataset[fi]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# metrics
		logs = model.evaluate(img_gt, denoised, device)
		for key in logs:
			ret[key].append(logs[key].cpu().numpy())

		for _ in range(args.iter): # comparison to F2F
			# pseudo noisy
			input_dict = {'noisy_input' : denoised.cpu()}
			input_dict['noisy_input'] = gaussian_noisy(input_dict['noisy_input'], SIGMA)
			pseudo_clean_frame = denoised

			model.train_step(pseudo_clean_frame, input_dict, device)
		
	return ret


def rfr_offline(gt_frames, noisy_dataset, ckpt_path, img_path, n_step, log_path=None):
	optimizer.state = collections.defaultdict(dict) # opt reinit

	frame_len = len(gt_frames)
	psuedo_clean_frames = []
	input_dicts = []

	for fi in range(frame_len):
		# noisy
		input_dict = noisy_dataset[fi]

		# denoised
		with torch.no_grad():
			denoised = torch.clamp(model(input_dict, device), 0., 1.)

		psuedo_clean_frames.append(denoised[0].cpu())
		input_dicts.append(input_dict)
		
	psuedo_clean_ds = get_dataset(psuedo_clean_frames)


	rets = []
	for step in range(n_step):
		print (step, datetime.datetime.now())
		# psuedo_clean_frames_ft = []
		# for fi in range(frame_len):
		# 	# noisy
		# 	input_dict = input_dicts[fi]

		# 	# denoised
		# 	with torch.no_grad():
		# 		denoised = torch.clamp(model(input_dict, device), 0., 1.)

		# 	psuedo_clean_frames_ft.append(denoised[0].cpu())
			

		# psuedo_clean_ft_ds = get_dataset(psuedo_clean_frames_ft)

		# random permute frames
		idxes = np.arange(frame_len)
		np.random.shuffle(idxes)
		for fi in idxes:
			# pseudo noisy
			input_dict = psuedo_clean_ds[fi]
			input_dict['noisy_input'] = gaussian_noisy(input_dict['noisy_input'], SIGMA)
			pseudo_clean_frame = psuedo_clean_frames[fi]

			# input_dict_ft = psuedo_clean_ft_ds[fi]
			# input_dict_ft['noisy_input'] = gaussian_noisy(input_dict_ft['noisy_input'], SIGMA)
			# pseudo_clean_frame_ft = psuedo_clean_frames_ft[fi]

			if REBUTTAL: # without second term loss
				# input_dict = input_dict_ft
				# pseudo_clean_frame = pseudo_clean_frame_ft
				pass

			else:
				# concat
				for key in input_dict:
					input_dict[key] = torch.cat((input_dict[key], input_dict_ft[key]), 0)

				pseudo_clean_frame = torch.stack((pseudo_clean_frame, pseudo_clean_frame_ft), 0)

			# finetune
			model.train_step(pseudo_clean_frame, input_dict, device)

		checkpoint_path = os.path.join(ckpt_path, '{:02}'.format(step + 1) + '.pt')
		# torch.save(denoiser.state_dict(), checkpoint_path)

		ret = evaluation(gt_frames, noisy_dataset)
		rets.append(ret)

		with open(os.path.join(log_path, 'rfr_offline.pkl'), 'wb') as outfile:
			pickle.dump(rets, outfile)

	return rets





if __name__ == '__main__':
	finetuning_mode()

	video_len = len(eval_gt_frame_paths)
		
	for vi in range(video_len):
		gt_frame_dir, gt_frame_names = eval_gt_frame_paths[vi]
		noisy_frame_dir, noisy_frame_names = eval_noisy_frame_paths[vi]
		_, video_name = get_folder_img_name(gt_frame_dir)

		if video_name != 'crowd_run_1080p50':#'girl-dog':# and video_name != 'salsa':
			continue

		print ('=========================', flush=True)
		print (video_name, flush=True)

		
		result_path = os.path.join('results', model.get_log_path(), args.test, video_name)
		if REBUTTAL:
			result_path = os.path.join('results', model.get_log_path(), args.test + '_rebuttal', video_name)
		if BLIND:
			result_path = os.path.join('results', model.get_log_path(), args.test + '_BLIND', video_name)
		if args.iter > 1:
			result_path = os.path.join('results', model.get_log_path(), args.test + '_{}'.format(args.iter), video_name)
		img_path = os.path.join(result_path, 'imgs')
		log_path = os.path.join(result_path, 'logs')
		ckpt_path = os.path.join(result_path, 'ckpts')
		make_dir(log_path)
		make_dir(ckpt_path)
		

		frame_len = len(gt_frame_names)

		# frames
		gt_frames, noisy_frames = [], []
		for fi in range(frame_len):
			gt_frames.append(transforms.ToTensor()(Image.open(os.path.join(gt_frame_dir, gt_frame_names[fi]))))
			noisy_frames.append(transforms.ToTensor()(Image.open(os.path.join(noisy_frame_dir, noisy_frame_names[fi]))))

		noisy_dataset = get_dataset(noisy_frames)


		if 1:#args.iter == 1: # pre-trained
			load_denoiser()

			print (video_name, 'evaluation')
			ret = evaluation(gt_frames, noisy_dataset, img_path)

			with open(os.path.join(log_path, 'evaluation.pkl'), 'wb') as outfile:
				pickle.dump(ret, outfile)

			with open(os.path.join(log_path, 'evaluation.pkl'), 'rb') as infile:
				load_params = pickle.load(infile)
				idx = 0
				for a, b in zip(load_params['psnr'], load_params['ssim']):
					print (idx, a, b)
					idx += 1


		if 0: # online
			load_denoiser()

			print (video_name, 'rfr_online')
			ret = rfr_online(gt_frames, noisy_dataset, img_path + '_online')

			with open(os.path.join(log_path, 'rfr_online.pkl'), 'wb') as outfile:
				pickle.dump(ret, outfile)

			with open(os.path.join(log_path, 'rfr_online.pkl'), 'rb') as infile:
				load_params = pickle.load(infile)
				idx = 0
				for a, b in zip(load_params['psnr'], load_params['ssim']):
					print (idx, a, b)
					idx += 1


		if 1: # offline:
			load_denoiser()

			print (video_name, 'rfr_offline')
			rets = rfr_offline(gt_frames, noisy_dataset, ckpt_path, img_path, 10, log_path)

			with open(os.path.join(log_path, 'rfr_offline.pkl'), 'wb') as outfile:
				pickle.dump(rets, outfile)

			with open(os.path.join(log_path, 'rfr_offline.pkl'), 'rb') as infile:
				load_params = pickle.load(infile)
				print (len(load_params))
				for i in range(len(load_params)):
					print (i, '============')
					for a, b in zip(load_params[i]['psnr'], load_params[i]['ssim']):
						print (a, b)