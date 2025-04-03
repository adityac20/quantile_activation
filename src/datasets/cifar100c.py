import logging
from glob import glob
from os import path as osp
from os.path import join as osj

import numpy as np
import torch

from PIL import Image
from torchvision import datasets
from torchvision import transforms


# distortions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise',
#     'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
#     'snow', 'frost', 'fog', 'brightness',
#     'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
#     'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
# ]

distortions_paper = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


list_distorions = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]
list_severity = [1, 2, 3, 4, 5]


class DistortedCIFAR100(datasets.VisionDataset):
    """
    In CIFAR-100-C, the first 10,000 images in each .npy are the test set images
    corrupted at severity 1, and the last 10,000 images are the test set images
    corrupted at severity five. labels.npy is the label file for all other image
    files.

    @article{hendrycks2019robustness,
      title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
      author={Hendrycks, Dan and Dietterich, Thomas},
      journal={Proceedings of the International Conference on Learning Representations},
      year={2019}
    }

    """

    list_distorions = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    list_severity = [1, 2, 3, 4, 5]

    def __init__(self, root, distortion, severity, transform, target_transform=None) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data = np.load(f"{root}/CIFAR-100-C/{distortion}.npy")
        self.targets = np.load(f"{root}/CIFAR-100-C/labels.npy")
        self.ind_start, self.ind_end = (severity - 1) * 10000, (severity) * 10000
        self.data_dir = root
        self.num_classes = 10

    def __getitem__(self, index):
        img, target = self.data[self.ind_start + index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return 10000
