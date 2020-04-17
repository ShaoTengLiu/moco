import torchvision.transforms as transforms
from corruption import (make_augmentation, gaussian_noise, gaussian_noise_con, pixelate, \
    gaussian_noise_05, contrast, fog)
import moco.loader

# stliu: change the number for CIFAR
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.247, 0.243, 0.261])
# stliu: only use v2
# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
# augmentation = [
# 	transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
# 	transforms.RandomApply([
# 		transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
# 	], p=0.8),
# 	transforms.RandomGrayscale(p=0.2),
# 	transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
# 	transforms.RandomHorizontalFlip(),
# 	transforms.ToTensor(),
# 	normalizenormalize
# ]
# stliu: CIFAR version
def aug(aug_type, level=5):
    if aug_type == 'original':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'v1':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'ttt':
        augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'gaussian_noise':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation(gaussian_noise, level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'gaussian_noise_05':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation(gaussian_noise_05, level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'contrast':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation(contrast, level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'fog':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation(fog, level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'gaussian_noise_conti':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation(gaussian_noise_con, level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'pixelate':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation(pixelate, level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'gaussian_noise,fog':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation([gaussian_noise,fog], level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == 'gaussian_noise,contrast':
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([make_augmentation([gaussian_noise,contrast], level)], p=0.5),
            transforms.ToTensor(),
            normalize
        ]

    return augmentation

test_transform = transforms.Compose([transforms.ToTensor(), normalize])