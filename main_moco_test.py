#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder

# stliu: my import
from ttt_resnet import ResNetCifar # stliu: to use tiny resnet from ttt
from util import save_checkpoint, AverageMeter, ProgressMeter, adjust_learning_rate, accuracy # stliu: they were defined in this file
import tensorboard_logger as tb_logger # stliu: to use tensorboard
from tqdm import tqdm
import liblinear_multicore.python.liblinearutil as liblinearutil
from ttt_utils.misc import *
from ttt_utils.test_helpers import *
from ttt_utils.prepare_dataset import *
from ttt_utils.rotation import rotate_batch
from PIL import Image
from transforms import *
from util import get_norm
import datetime
from detectron2_utils.batch_norm import FrozenBatchNorm2d

best_acc1 = 0

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))
model_names.append('resnet_ttt') # stliu: use small ResNet from ttt

def parse_option(): # design a function for parse
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
	parser.add_argument('data', metavar='DIR',
						help='path to dataset')
	parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
						choices=model_names,
						help='model architecture: ' +
							' | '.join(model_names) +
							' (default: resnet50)')
	parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
						help='number of data loading workers (default: 32)')
	parser.add_argument('--epochs', default=200, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
						help='manual epoch number (useful on restarts)')
	parser.add_argument('-b', '--batch-size', default=256, type=int,
						metavar='N',
						help='mini-batch size (default: 256), this is the total '
							'batch size of all GPUs on the current node when '
							'using Data Parallel or Distributed Data Parallel')
	parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
						metavar='LR', help='initial learning rate', dest='lr')
	parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
						help='learning rate schedule (when to drop lr by 10x)')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum of SGD solver')
	parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)',
						dest='weight_decay')
	parser.add_argument('-p', '--print-freq', default=10, type=int,
						metavar='N', help='print frequency (default: 10)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('--world-size', default=-1, type=int,
						help='number of nodes for distributed training')
	parser.add_argument('--rank', default=-1, type=int,
						help='node rank for distributed training')
	parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
						help='url used to set up distributed training')
	parser.add_argument('--dist-backend', default='nccl', type=str,
						help='distributed backend')
	parser.add_argument('--seed', default=None, type=int,
						help='seed for initializing training. ')
	parser.add_argument('--gpu', default=None, type=int,
						help='GPU id to use.')
	parser.add_argument('--multiprocessing-distributed', action='store_true',
						help='Use multi-processing distributed training to launch '
							'N processes per node, which has N GPUs. This is the '
							'fastest way to use PyTorch for either single node or '
							'multi node data parallel training')

	# moco specific configs:
	parser.add_argument('--moco-dim', default=128, type=int,
						help='feature dimension (default: 128)')
	parser.add_argument('--moco-k', default=65536, type=int,
						help='queue size; number of negative keys (default: 65536)')
	parser.add_argument('--moco-m', default=0.999, type=float,
						help='moco momentum of updating key encoder (default: 0.999)')
	parser.add_argument('--moco-t', default=0.07, type=float,
						help='softmax temperature (default: 0.07)')

	# options for moco v2
	parser.add_argument('--mlp', action='store_true',
						help='use mlp head')
	parser.add_argument('--aug-plus', action='store_true',
						help='use moco v2 data augmentation')
	parser.add_argument('--cos', action='store_true',
						help='use cosine lr schedule')

	# stliu: new options
	parser.add_argument('--model_path', default='./', help='the folder to save models')
	parser.add_argument('--tb_path', default='./', help='the folder to save tensorboards')
	parser.add_argument('--width', type=int, default=1, help='the width of ResNet(resnet_ttt)')
	parser.add_argument('-s', '--save-freq', default=10, type=int,
						metavar='N', help='save frequency (default: 10)')
	parser.add_argument('--svm-freq', default=10, type=int,
						metavar='N', help='SVM frequency (default: 10)')
	parser.add_argument('--dataset', default='cifar10')
	parser.add_argument('--depth', type=int, default=26, help='the depth of ResNet(resnet_ttt)')
	parser.add_argument('--shared', default=None)
	parser.add_argument('--rotation_type', default='rand')
	parser.add_argument('--val', default=None)
	parser.add_argument('--ttt', action='store_true')
	parser.add_argument('--aug', default='original')
	parser.add_argument('--norm', default='bn')
	parser.add_argument('--frozen', action='store_true') # freeze the norm(bn) when ttt
	parser.add_argument('--bn_only', action='store_true') # only update bn at test time


	# stliu: one can do something with parsers here
	opt = parser.parse_args()
	opt.model_name = 'moco_ttt_{}_w{}_{}_lr_{}_bsz_{}_k_{}_t_{}_{}'.format(opt.norm, opt.width, opt.arch, opt.lr, opt.batch_size, opt.moco_k, opt.moco_t, opt.aug)
	opt.model_folder = os.path.join(opt.model_path, opt.model_name)
	opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
	if not os.path.isdir(opt.model_folder):
		os.makedirs(opt.model_folder)
	if not os.path.isdir(opt.tb_folder):
		os.makedirs(opt.tb_folder)
	
	return opt

# stliu: create a function for dataloader
def get_loader(args):
	# Data loading code
	# stliu: change to CIFAR
	if ',' in args.aug:
		train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, \
			transform=moco.loader.TwoCropsTransform(transforms.Compose(aug(args.aug.split(',')[0], int(args.aug.split(',')[1])))))
	else:
		train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True, \
			transform=moco.loader.TwoCropsTransform(transforms.Compose(aug(args.aug))))
	# stliu: change the above Data loading code into CIFAR-10 version

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

	# stliu: add two other loader for SVM
	memory_data = datasets.CIFAR10(root=args.data, train=True, download=True, transform=test_transform)
	memory_loader = torch.utils.data.DataLoader(memory_data, batch_size=args.batch_size, 
								shuffle=False, num_workers=args.workers, pin_memory=True)
	test_data = datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
	if args.val and args.val != 'original':
		corruption, level = args.val.split(',')
		teset_raw = np.load(args.data + '/CIFAR-10-C-trainval/val/%s.npy' %(corruption))[(int(level)-1)*10000: int(level)*10000]
		test_data.data = teset_raw
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
								shuffle=False, num_workers=args.workers, pin_memory=True)
	return train_loader, train_sampler, memory_loader, test_loader, test_data

# stliu: the order of functions has been changed
def train(train_loader, model, criterion, optimizer, epoch, args, ssh):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	# progress = ProgressMeter(
	# 	len(train_loader),
	# 	[batch_time, data_time, losses, top1, top5],
	# 	prefix="Epoch: [{}]".format(epoch))

	# stliu: design new pregress
	epoch_time = AverageMeter('Epoch Time', ':6.3f')
	progress = ProgressMeter(
		len(train_loader),
		[epoch_time, losses, top1, top5],
		prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	if args.norm != 'bnf':
		model.train()
		ssh.train()

	end = time.time()

	for i, (images, _) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		if args.gpu is not None:
			images[0] = images[0].cuda(args.gpu, non_blocking=True)
			images[1] = images[1].cuda(args.gpu, non_blocking=True)

		# compute output
		output, target = model(im_q=images[0], im_k=images[1])
		print(output.size())
		loss = criterion(output, target)
		if args.shared is not None:
			inputs_ssh, labels_ssh = rotate_batch(images[0], args.rotation_type)
			inputs_ssh, labels_ssh = inputs_ssh.cuda(args.gpu, non_blocking=True), labels_ssh.cuda(args.gpu, non_blocking=True)
			outputs_ssh = ssh(inputs_ssh)
			loss_ssh = criterion(outputs_ssh, labels_ssh)
			loss += loss_ssh
		# acc1/acc5 are (K+1)-way contrast classifier accuracy
		# measure accuracy and record lossa
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), images[0].size(0))
		top1.update(acc1[0], images[0].size(0))
		top5.update(acc5[0], images[0].size(0))
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		epoch_time.update(batch_time.avg*len(train_loader))
		end = time.time()

		if (i+1) % args.print_freq == 0: # stliu: change i to i+1
			progress.display(i)
	
	return losses.avg

# stliu: use SVM to test model
def test(train_loader, model_kq, model, val_loader, config_lsvm, args, ssh):
	err_ssh = 0 if args.shared is None else test_ttt(val_loader, ssh, sslabel='expand')[0]
	print('SSH ERROR:', err_ssh)

	model.eval()
	top1, feats_bank = AverageMeter('Acc@1', ':4.2f'), []

	with torch.no_grad():
		# generate feature bank
		for (images, _) in tqdm(train_loader, desc='Feature extracting'):
			feats= model(images.cuda(args.gpu, non_blocking=True), 'r')
			feats_bank.append(feats)
	feats_bank = torch.cat(feats_bank, dim=0)
	label_bank = torch.tensor(train_loader.dataset.targets)
	model_lsvm = liblinearutil.train(label_bank.cpu().numpy(), feats_bank.cpu().numpy(), config_lsvm)

	with torch.no_grad():
		val_bar = tqdm(val_loader)
		for (images, target) in val_bar:
			images = images.cuda(args.gpu, non_blocking=True)

			# compute output
			feats = model(images, 'r')
			_, top1_acc, _ = liblinearutil.predict(target.cpu().numpy(), feats.cpu().numpy(), model_lsvm, '-q')

			# measure accuracy and record
			top1.update(top1_acc[0], images.size(0))
			val_bar.set_description('Acc@SVM:{:.2f}%'.format(top1.avg))
	return top1.avg

# stliu: use SVM to test model
def ttt_test(train_loader, model_kq, model, val_loader, config_lsvm, args, ssh, teset, head):
	if ',' in args.aug:
		tr_transform = transforms.Compose(aug(args.aug.split(',')[0], int(args.aug.split(',')[1])))
	else:
		tr_transform = transforms.Compose(aug(args.aug))
	err_ssh = 0 if args.shared is None else test_ttt(val_loader, ssh, sslabel='expand')[0]
	print('SSH ERROR:', err_ssh)

	model.eval()
	top1, feats_bank = AverageMeter('Acc@1', ':4.2f'), []

	with torch.no_grad():
		# generate feature bank
		for (images, _) in tqdm(train_loader, desc='Feature extracting'):
			feats= model(images.cuda(args.gpu, non_blocking=True), 'r')
			feats_bank.append(feats)
	feats_bank = torch.cat(feats_bank, dim=0)
	label_bank = torch.tensor(train_loader.dataset.targets)
	model_lsvm = liblinearutil.train(label_bank.cpu().numpy(), feats_bank.cpu().numpy(), config_lsvm)

	# stliu: ttt
	model.train()
	val_bar = tqdm(val_loader)
	for (images, _) in val_bar:
		images = images.cuda(args.gpu, non_blocking=True)
		# update BN
		feats = model(images, 'r')
		val_bar.set_description('Test-time Forward')

	model.eval()
	with torch.no_grad():
		val_bar = tqdm(val_loader)
		for (images, target) in val_bar:
			images = images.cuda(args.gpu, non_blocking=True)

			# compute output
			feats = model(images, 'r')
			_, top1_acc, _ = liblinearutil.predict(target.cpu().numpy(), feats.cpu().numpy(), model_lsvm, '-q')

			# measure accuracy and record
			top1.update(top1_acc[0], images.size(0))
			val_bar.set_description('Acc@SVM:{:.2f}%'.format(top1.avg))
	return top1.avg

def main_worker(gpu, ngpus_per_node, args):
	global best_acc1 # stliu: best accuracy
	args.gpu = gpu

	# suppress printing if not master
	if args.multiprocessing_distributed and args.gpu != 0:
		def print_pass(*args):
			pass
		builtins.print = print_pass

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
	# create model
	print("=> creating model '{}'".format(args.arch))
	# stliu: add resnet_ttt
	if args.arch == 'resnet_ttt':
		model = moco.builder.MoCo(
			ResNetCifar,
			args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, width=args.width, norm=args.norm)
		_, ext, head, ssh = build_model(args, model.encoder_q) # stliu: ext, head and ssh share same paras as encoder_q
		# stliu: SVM with model_val on single GPU
		norm_layer = get_norm(args.norm)
		model_val = ResNetCifar(num_classes=args.moco_dim, width=args.width, norm_layer=norm_layer)
	else:
		model = moco.builder.MoCo(
			models.__dict__[args.arch],
			args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
	# print(model) # stliu: comment this

	if args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			model_val.cuda(args.gpu) # stliu: for SVM
			ssh = ssh.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			# stliu: add broadcast_buffers=False to use normal BN
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
			ssh = torch.nn.parallel.DistributedDataParallel(ssh, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
			# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
			# ssh = torch.nn.parallel.DistributedDataParallel(ssh, device_ids=[args.gpu])
		else:
			model.cuda()
			model_val.cuda() # stliu: for SVM
			ssh = ssh.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
			ssh = torch.nn.parallel.DistributedDataParallel(ssh, broadcast_buffers=False, find_unused_parameters=True)
			# model = torch.nn.parallel.DistributedDataParallel(model)
			# ssh = torch.nn.parallel.DistributedDataParallel(ssh)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
		model_val = model_val.cuda(args.gpu) # stliu: for SVM
		ssh = ssh.cuda(args.gpu)
		# comment out the following line for debugging
		raise NotImplementedError("Only DistributedDataParallel is supported.")
	else:
		# AllGather implementation (batch shuffle, queue update, etc.) in
		# this code only supports DistributedDataParallel.
		raise NotImplementedError("Only DistributedDataParallel is supported.")

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)

	parameters = list(model.parameters())+list(head.parameters())
	optimizer = torch.optim.SGD(parameters, args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			args.start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			head.load_state_dict(checkpoint['head'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	# stliu: I design it as a function
	train_loader, train_sampler, memory_loader, test_loader, teset = get_loader(args)

	if args.val:
		state_dict = model.state_dict()
		for k in list(state_dict.keys()):
			if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
				state_dict[k[len("module.encoder_q."):]] = state_dict[k]
			del state_dict[k]
		model_val.load_state_dict(state_dict, strict=False)
		flag_liblinear = '-s 2 -q -n '+str(args.workers)
		if args.ttt:
			test_acc_svm = ttt_test(memory_loader, model, model_val, test_loader, flag_liblinear, args, ssh, teset, head)
		else:
			test_acc_svm = test(memory_loader, model, model_val, test_loader, flag_liblinear, args, ssh)
		print('#### result ####\n' + args.val + ':', test_acc_svm, '\n################')
	else:
		# stliu: tensorboard
		logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

		for epoch in range(args.start_epoch, args.epochs):
			if args.distributed:
				train_sampler.set_epoch(epoch)
			adjust_learning_rate(optimizer, epoch, args)

			# train for one epoch
			loss = train(train_loader, model, criterion, optimizer, epoch, args, ssh)

			# stliu: tensorboard logger
			logger.log_value('loss', loss, epoch)

			if not args.multiprocessing_distributed or (args.multiprocessing_distributed
					and args.rank % ngpus_per_node == 0):
				if epoch % args.save_freq == 0 and epoch != 0: # stliu: ignore the first model
					print('==> Saving...')
					save_checkpoint({
						'epoch': epoch + 1,
						'arch': args.arch,
						'state_dict': model.state_dict(),
						'head': head.state_dict(),
						'optimizer' : optimizer.state_dict(),
					}, is_best=False, filename=args.model_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch))
			# stliu: test with SVM
			if (epoch+1) % args.svm_freq == 0:
				state_dict = model.state_dict()
				for k in list(state_dict.keys()):
					if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
						state_dict[k[len("module.encoder_q."):]] = state_dict[k]
					del state_dict[k]
				model_val.load_state_dict(state_dict, strict=False)
				flag_liblinear = '-s 2 -q -n '+str(args.workers)
				test_acc_svm = test(memory_loader, model, model_val, test_loader, flag_liblinear, args, ssh)
				
				# stliu: save the best model
				is_best = test_acc_svm > best_acc1
				best_acc1 = max(test_acc_svm, best_acc1)
				if is_best:
					print('==> Saving the Best...')
					save_checkpoint({
						'epoch': epoch + 1,
						'arch': args.arch,
						'state_dict': model.state_dict(),
						'head': head.state_dict(),
						'optimizer' : optimizer.state_dict(),
					}, is_best=True, filename=args.model_folder + '/best.pth.tar'.format(epoch))
		print('The Best SVM Accuracy:', best_acc1)

def main():
	args = parse_option()# stliu: use a function

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					  'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
	main()