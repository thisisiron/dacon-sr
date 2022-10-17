from typing import Optional 

import random
import numpy as np
from contextlib import contextmanager

import torch
from torch import optim
import torch.distributed as dist
from torch.optim import lr_scheduler

from configuration.const import logger


def get_optimizer(param, opt_name: str, lr: float, weight_decay: float):
    if opt_name == 'SGD':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'Adam':
        optimizer = optim.Adam(param, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif opt_name == 'AdamW':
        optimizer = optim.AdamW(param, lr=lr, betas=(0.9, 0.999))  # adjust beta1 to momentum
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(param, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError('The optimizer should be in [SGD, AdamP, ...]')
    return optimizer


def get_scheduler(optimizer, scheduler_name: str, opt: dict):
    if scheduler_name == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, threshold=0.01, patience=5)
    elif scheduler_name == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=0)  # TODO: set T_max
    else:
        raise NotImplementedError(f'{scheduler_name} is not implemented')

    return scheduler


def print_log(log: str, status: dict):
    for name, val in status.items():
        log += f' | {name}: {val.avg:.4f}'
    return log
    # logger.info(log)


def seed_everything(seed=0):
    """
    Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible

    Args:
        seed: integer
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True  # side effect
    np.random.seed(seed)
    random.seed(seed)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AverageMeter:
    def __init__(self):
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


def mu_tonemap(img):
    MU = 5000.0
    return torch.log(1.0 + MU * (img + 1.0) / 2.0) / np.log(1.0 + MU)


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
