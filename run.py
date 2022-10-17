import os
import time
from tqdm import tqdm
from collections import defaultdict

import cv2
import numpy as np
import torch

from configuration.const import logger
from metrics import cal_psnr
from losses import tv_loss
from utils import print_log
from utils import mu_tonemap
from utils import AverageMeter

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def iteration(opt, epoch, generator, criterion,
          g_optimizer, train_loader, log_dir, train, device):

    status = defaultdict(AverageMeter) 

    pbar = enumerate(train_loader, 1)
    if RANK in {-1, 0}:
        pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader))

    for i, (lr_imgs, hr_imgs) in pbar:
        start = time.time()

        loss_gan = torch.zeros(1, device=device)
        loss_vgg = torch.zeros(1, device=device)
        loss_rec = torch.zeros(1, device=device)
        loss_l2 = torch.zeros(1, device=device)
        loss_tv = torch.zeros(1, device=device)
        loss_g = torch.zeros(1, device=device)
        loss_fm = torch.zeros(1, device=device)

        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        
        sr_imgs = generator(lr_imgs)

        loss_rec = criterion['l1'](sr_imgs, hr_imgs) * opt.lamb_l1
        status['l1_loss'].update(loss_rec.item())

        if opt.lamb_l2 > 0:
            loss_l2 = criterion['l2'](sr_imgs, hr_imgs) * opt.lamb_l2
            status['l2_loss'].update(loss_l2.item())

        if opt.lamb_percep > 0:
            mu_tonemap_gt = mu_tonemap(hr_imgs)

            loss_vgg = criterion['perceptual_loss'](mu_tonemap(sr_imgs), mu_tonemap_gt) * opt.lamb_percep
            status['perceptual loss'].update(loss_vgg.item())

        if opt.lamb_tv > 0:
            loss_tv = tv_loss(sr_imgs.detach(), opt.lamb_tv)
            status['tv loss'].update(loss_tv.item())

        loss_g = loss_rec + loss_vgg + loss_gan  + loss_tv + loss_fm + loss_l2

        if train:
            g_optimizer.zero_grad()
            loss_g.backward()
            g_optimizer.step()

        if RANK in {-1, 0}:

            status['PSNR'].update(cal_psnr(sr_imgs.add(1).mul(127.5).detach(), hr_imgs.add(1).mul(127.5).detach()))

            if i % 1000 == 0:  # print every 100 mini-batches and save images
                image_dict: dict = {}
                image_dict['lr_imgs'] = lr_imgs.detach() 
                image_dict['hr_imgs'] = hr_imgs.detach()
                image_dict['fake_a'] = sr_imgs.detach()

                log = f"step: {i + (epoch - 1) * len(train_loader)} | time: {time.time() - start:.4f} sec"
                log = print_log(log, status)

                pbar.set_description(log, refresh=True) 

    return status 
