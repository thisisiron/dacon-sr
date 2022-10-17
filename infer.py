import os
import sys
from pathlib import Path
import cv2
import json
import zipfile
from glob import glob
from tqdm.auto import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse

import numpy as np
import torch
from torch import nn

import segmentation_models_pytorch as smp

from networks import select_generator
from networks import load_weight
from utils import dotdict


def inference(generator, test_dir, opt, batch_size, img_size, stride):
    """
    :param model: model
    :param test_dir: test path
    :return:
    pandas.DataFrame: columns should be include "image_name" and "y_pred".
    """

    image_paths = sorted(glob(os.path.join(test_dir, '*')))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = []

    with torch.no_grad():
        pbar = tqdm(enumerate(image_paths), total=len(image_paths))
        for i, img_path in pbar:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (2048 // opt.scale, 2048 // opt.scale))  # check

            assert ((img.shape[0] / stride) * (img.shape[1] / stride)) % batch_size == 0

            crop = []
            position = []
            batch_count = 0

            result_img = np.zeros((2048, 2048, 3)).astype(np.float32)
            voting_mask = np.zeros((2048, 2048, 3)).astype(np.float32)

            img = img / 127.5 - 1

            num_patch = 0

            for top in range(0, img.shape[0], stride):
                for left in range(0, img.shape[1], stride):

                    # piece = np.zeros([img_size, img_size, 3], np.float32)
                    temp = img[top:top + img_size // opt.scale, left:left + img_size // opt.scale, :]
                    # piece[:temp.shape[0], :temp.shape[1], :] = temp
                    crop.append(temp)
                    position.append([top * opt.scale, left * opt.scale])
                    batch_count += 1
                    if batch_count == batch_size:
                        crop = torch.Tensor(np.array(crop)).to(device).permute(0, 3, 1, 2)
                        generated = generator(crop)
                        generated.clamp_(-1, 1).add_(1).mul_(127.5)
                        generated = generated.permute(0, 2, 3, 1).detach().cpu().numpy()

                        crop = []
                        batch_count = 0
                        for num, (t, l) in enumerate(position):
                            piece = generated[num]
                            h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                            result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w]
                            voting_mask[t:t+img_size, l:l+img_size, :] += 1.
                        position = []
                    num_patch += 1

            pbar.set_description(f'The number of patch: {num_patch}')

            result_img = result_img / voting_mask
            result_img = result_img.astype(np.uint8)
            results.append(result_img)
            if batch_count != 0:
                print('batch_count!!! --->', batch_count)

    return results


def make_submission(result, image_size, stride, ckpt, exp_name='./'):
    save_path = os.path.join(ROOT / 'experiments' / exp_name, 'submission')
    os.makedirs(os.path.join(save_path), exist_ok=True)
    os.chdir(save_path)

    sub_imgs = []
    for i, img in enumerate(result):
        img_name = f'{20000+i}.png'
        cv2.imwrite(img_name, img)
        sub_imgs.append(img_name)

    submission = zipfile.ZipFile(f"submission_img{image_size}_s{stride}_{exp_name}-{ckpt}.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()
    print('END')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, help='model.pt path(s)')
    parser.add_argument('--batch', type=int, default=1, help='batch size')

    parser.add_argument('--img-size', type=int, default=2048, help='inference size (pixels)')
    parser.add_argument('--stride', '-s', type=int, default=512, help='stride size (pixels)')
    parser.add_argument('--data', nargs='+', type=str, default=ROOT / 'test/lr', help='testset path')

    parser.add_argument('--name', default='exp', help='save to project/name')
    args = parser.parse_args()

    return args 


def main():
    args = parse_args()
    path = args.weight.split('/')
    exp_name, ckpt = path[-4], path[-2].split('_')[-1]  # experiments/EXP_NAME/weight/ckpt_NUMBER/gene
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(ROOT / 'experiments' / exp_name, 'opt.json'), 'r') as f:
        opt = json.load(f)
    opt = dotdict(opt)

    if opt.gene in {'swinir', 'convnext'}:
        opt.scale = 4
    else:
        opt.scale = 1
    
    print(opt)
    opt.weight = args.weight

    generator = select_generator(opt, device, -1)

    # generator = smp.Unet(encoder_name=opt.enc,
    #                              in_channels=3,  
    #                              classes=3,                      
    #                              activation='tanh'
    #                             ).to(device)

    generator.eval()

    results = inference(generator, 
                        test_dir=args.data, 
                        opt=opt,
                        batch_size=args.batch,
                        img_size=args.img_size, 
                        stride=args.stride)

    make_submission(results, args.img_size, args.stride, ckpt, exp_name)


if __name__ == '__main__':
    main()
