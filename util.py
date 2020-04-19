# stliu: all functions are moved from main_moco.py

import math
import torch
import shutil

import torch.nn as nn
from detectron2_utils.batch_norm import FrozenBatchNorm2d, NaiveSyncBatchNorm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
	"""Decay the learning rate based on schedule"""
	lr = args.lr
	if args.cos:  # cosine lr schedule
		lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
	else:  # stepwise lr schedule
		for milestone in args.schedule:
			lr *= 0.1 if epoch >= milestone else 1.
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

import torch.nn.functional as F
# stliu: can be used easily in DDP
class my_BatchNorm2d(nn.BatchNorm2d):
	def frozen(self):
		self.track_running_stats=False

def get_norm(norm):
	if norm == 'bn':
		norm_layer = nn.BatchNorm2d
	elif norm == 'bnf':
		norm_layer = FrozenBatchNorm2d
	elif norm == 'sync':
		norm_layer = nn.SyncBatchNorm
	elif norm == 'nsync':
		norm_layer = NaiveSyncBatchNorm
	else:
		group_norm = int(norm.split(',')[1])
		def gn_helper(planes):
			return nn.GroupNorm(group_norm, planes)
		norm_layer = gn_helper
	return norm_layer

def set_bn_eval(m):
    if isinstance(m, my_BatchNorm2d):
        m.frozen()