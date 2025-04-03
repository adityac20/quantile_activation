import numpy as np

import torch

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from torchvision.transforms.v2 import AutoAugmentPolicy


def get_cifar100(root):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)

    testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)

    return trainset, testset, transform_train, transform_test
