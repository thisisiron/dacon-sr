import os
import PIL
import cv2
from glob import glob

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, distributed

from transform import get_transforms
from configuration.const import IMG_EXTENSIONS


DEVICE_COUNT = max(torch.cuda.device_count(), 1)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def create_dataloader(opt, batch_size, shuffle=False, rank=-1):
        train_transform = get_transforms(augment_type=opt.augment_type, 
                                         image_norm=opt.image_norm)

        dataset = TrainDataset(image_dir=opt.data, 
                               hr_dir=opt.data_hr if opt.data_hr else opt.data,
                               transform=train_transform)
        batch_size = min(batch_size, len(dataset))
        nw = min([os.cpu_count() // DEVICE_COUNT, 
                  batch_size if batch_size > 1 else 0, opt.num_workers])  # number of workers
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)

        loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle and sampler is None,
                            num_workers=nw,
                            sampler=sampler,
                            pin_memory=True)
        return loader


class TrainDataset(Dataset):
    def __init__(self, image_dir, hr_dir, transform=None):
        self.low_paths = sorted(glob(os.path.join('data/train', 'lr', image_dir, '*')))
        self.high_paths = sorted(glob(os.path.join('data/train', 'hr', hr_dir, '*')))
        assert len(self.low_paths) == len(self.high_paths), f"{len(self.low_paths)} != {len(self.high_paths)}"
        self.transform = transform
        print('image count', len(self.low_paths))

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        lr_img = np.load(self.low_paths[idx])
        hr_img = np.load(self.high_paths[idx])

        if self.transform:
            transformed = self.transform(image=lr_img, image2=hr_img)
            lr_img = transformed['image']
            hr_img = transformed['image2']
            lr_img = np.transpose(lr_img, (2, 0, 1)).astype(np.float32) 
            hr_img = np.transpose(hr_img, (2, 0, 1)).astype(np.float32) 
        
        return lr_img, hr_img 
