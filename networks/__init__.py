import os
import torch

import functools
import yaml

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils import model_zoo

from networks.swinir import SwinIR
from networks.convnext import ConvNeXt
from networks.convnext import model_urls

import segmentation_models_pytorch as smp

from utils import dotdict


def save_model(model, filename):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)


def load_weight(model, weight_file, device):
    """Load trained weight.
    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file, map_location=device), strict=True)
        # model.load_state_dict(torch.load(weight_file), strict=False)
        print('load weight from {}.'.format(weight_file))
    else:
        raise f'weight file {weight_file} is not exist.'


def select_generator(opt, device, rank):

    if opt.cfg:
        with open(opt.cfg, 'r') as f:
            cfg = dotdict(yaml.safe_load(f))

    if opt.gene == 'unet':
        generator = smp.Unet(encoder_name=opt.enc,
                             encoder_weights="imagenet",  
                             in_channels=3,  
                             classes=3,                      
                             activation='tanh'
                            )
    elif opt.gene == 'unet++':
        generator = smp.UnetPlusPlus(encoder_name=opt.enc,
                                     encoder_weights="imagenet",  
                                     in_channels=3,  
                                     classes=3,                      
                                     activation=opt.activation,
                                    )
    elif opt.gene == 'pan':
        generator = smp.PAN(encoder_name=opt.enc,
                            encoder_weights="imagenet",  
                            in_channels=3,  
                            classes=3,                      
                            activation=opt.activation,
                           )
    elif opt.gene == 'fpn':
        generator = smp.FPN(encoder_name=opt.enc,
                            encoder_weights="imagenet",  
                            in_channels=3,  
                            classes=3,                      
                            activation=opt.activation,
                           )
    elif opt.gene == 'deeplabv3+':
        generator = smp.DeepLabV3Plus(encoder_name=opt.enc,
                                      encoder_weights="imagenet",  
                                      in_channels=3,  
                                      classes=3,                      
                                      activation=opt.activation,
                                     )
    elif opt.gene == 'swinir':
        generator = SwinIR(in_chans=3,
                           upscale=cfg.upscale,
                           img_siz=cfg.image_size,
                           window_size=cfg.window_size,
                           depths=cfg.depths,
                           embed_dim=cfg.embed_dim,
                           num_heads=cfg.num_heads,
                           mlp_ratio=cfg.mlp_ratio,
                           upsampler=cfg.upsampler,
                           activation=opt.activation,
                           resi_connection=cfg.resi_connection)
    elif opt.gene == 'convnext':
        generator = ConvNeXt(in_chans=3,
                             depths=[6, 6, 6, 6], 
                             # depths=[3, 3, 27, 3], 
                             dims=[180, 180, 180, 180],
                             )

    if opt.weight is not None:
        load_weight(generator, opt.weight, device)

    return generator.to(device)
