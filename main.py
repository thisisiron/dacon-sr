import os
import cv2
import zipfile
import time
import json
import argparse
from datetime import datetime

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from configuration.const import logger
from run import iteration 
from networks import select_generator
from networks import load_weight
from networks import save_model
from losses import VGGLoss
from dataset import create_dataloader

from utils import get_optimizer, get_scheduler
from utils import print_log
from utils import seed_everything
from utils import get_rank

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train_process(opt, generator, criterion,
                  g_optimizer, g_scheduler,
                  train_loader, val_loader, log_dir, device):
    save_epoch = 1 

    for epoch in range(1, opt.num_epoch + 1):
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        ### Train Process ### 
        total_start = time.time()
        generator.train()

        train_status = iteration(opt, epoch, generator, criterion,
                                 g_optimizer, train_loader, log_dir, train=True, device=device)

        g_scheduler.step()

        if RANK in {-1, 0}:
            logger.info(f'Learning rate(G) annealed to : {g_optimizer.param_groups[0]["lr"]:.6f} @epoch{epoch}')

        if RANK in {-1, 0}:
            minutes, seconds = divmod(time.time() - total_start, 60)
            log = f">>> [Train] Epoch: {epoch}/{opt.num_epoch} | Time: {int(minutes):2d} min {seconds:.4f} sec"
            print_log(log, train_status)

        if RANK in {-1, 0}:
            # Saving model
            if epoch % save_epoch == 0:
                logger.info(f'[{epoch}] Save the model!')
                checkpoint = f'ckpt_{epoch}'
                os.makedirs(os.path.join(log_dir, 'weight', checkpoint))

                filename = os.path.join(log_dir, 'weight', checkpoint, 'gene')
                save_model(generator, filename)


def get_model_config():
    # Argument Settings
    parser = argparse.ArgumentParser(description='Pytorch Image Template')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--data', default='512x512_r2048x2048_s128x128', type=str, help='Path of dataset')
    parser.add_argument('--data_hr', default='', type=str, help='Path of HR image')
    parser.add_argument('--cfg', default='' , type=str, help='Path of model config')

    parser.add_argument('--seed', default=2022, type=int, help='Random seed')

    parser.add_argument('--num_workers', default=8, type=int, help='The number of workers')

    parser.add_argument('--augment_type', default='default', type=str, help='[default, color]')
    parser.add_argument('--image_norm', default='zero', type=str, help='[imagenet, zero]')

    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--num_epoch', default=200, type=int, help='The number of epochs')

    parser.add_argument('--gene', default='unet', type=str, help='Type of generator')
    parser.add_argument('--enc', default='resnext101_32x8d', type=str, help='Type of encoder')
    parser.add_argument('--activation', default=None, type=str, help='Type of activation')

    parser.add_argument('--lamb_l1', default=10.0, type=float)
    parser.add_argument('--lamb_l2', default=0.0, type=float)

    parser.add_argument('--lamb_percep', default=0.0, type=float)
    parser.add_argument('--lamb_tv', default=0.0, type=float)
    parser.add_argument('--lamb_fm', default=0.0, type=float)

    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['SGD', 'Adam', 'AdamW', 'RMSprop'])
    parser.add_argument('--scheduler', default='step', type=str, help='[plateau, cosine, step]')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--lr_decay_iters', default=50, type=float)

    parser.add_argument('--weight', type=str)

    return parser.parse_args()


def main():
    opt = get_model_config()
    opt.lmd = [0.005, 0.1, 0.1, 0.1, 0.1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if LOCAL_RANK != -1:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        torch.distributed.barrier()

    if RANK in {-1, 0}:
        print(opt)

    seed_everything(opt.seed + get_rank())
    
    # Model setting
    logger.info('Build Model')
    generator = select_generator(opt, device, LOCAL_RANK)

    total_param = sum([p.numel() for p in generator.parameters()])
    logger.info(f'Generator size: {total_param} tensors')

    if torch.cuda.device_count() > 1 and RANK == -1:
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        generator = DataParallel(generator)

    if RANK != -1:
        generator = DDP(generator, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        print('gene', next(generator.parameters()).device )

    log_dir = None
    if RANK in {-1, 0}:
        dirname = datetime.now().strftime("%m%d%H%M") + f'_{opt.name}'
        log_dir = os.path.join('./experiments', dirname)
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f'LOG DIR: {log_dir}')

    # Dataset setting
    logger.info('Set the dataset')
    train_loader = create_dataloader(opt, batch_size=opt.batch, shuffle=True, rank=LOCAL_RANK)
    val_loader = None

    # Loss setting
    criterion = {}
    criterion['l1'] = torch.nn.L1Loss().to(device)

    if opt.lamb_l2 > 0:
        logger.info('+ L2 Loss')
        criterion['l2'] = torch.nn.MSELoss().to(device)

    if opt.lamb_percep > 0:
        criterion['perceptual_loss'] = VGGLoss()

    # Optimizer setting
    g_optimizer = get_optimizer(generator.parameters(), opt.optimizer, opt.lr, opt.weight_decay)
    logger.info(f'Initial Learning rate(G): {g_optimizer.param_groups[0]["lr"]:.6f}')

    # Scheduler setting
    g_scheduler = get_scheduler(g_optimizer, opt.scheduler, opt)

    if RANK in {-1, 0}:

        # Saving Argumens
        with open(os.path.join(log_dir, 'opt.json'), 'w') as f:
            json.dump(opt.__dict__, f, indent=4, sort_keys=True)

    logger.info('Start to train!')
    train_process(opt, generator, criterion,
                  g_optimizer, g_scheduler,
                  train_loader=train_loader, val_loader=val_loader,
                  log_dir=log_dir, device=device)

    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
