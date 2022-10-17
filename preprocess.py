import os
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
import cv2


def cut_img(img_path_list, save_path, patch_h_size, patch_w_size, h_resize, w_reszie, h_stride, w_stride):
    os.makedirs(f'{save_path}/p{patch_h_size}x{patch_w_size}_r{h_resize}x{w_reszie}_s{h_stride}x{w_stride}', exist_ok=True)
    num = 0
    for path in tqdm(img_path_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (w_reszie, h_resize))
        name = path.split('/')[-1].split('.')[0]
        patch = 0
        for top in range(0, img.shape[0], h_stride):
            for left in range(0, img.shape[1], w_stride):
                piece = np.zeros([patch_h_size, patch_w_size, 3], np.uint8)
                temp = img[top:top+patch_h_size, left:left+patch_w_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save(f'{save_path}/p{patch_h_size}x{patch_w_size}_r{h_resize}x{w_reszie}_s{h_stride}x{w_stride}/{name}_{str(patch).zfill(5)}.npy', piece)
                patch += 1
        num+=1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--patch_size', nargs='+', type=int, default=[512, 512], help='image size')
    parser.add_argument('--resize', nargs='+', type=int, default=[2048, 2048], help='resize')
    parser.add_argument('--stride', nargs='+', type=int, default=[128, 128], help='stride size')
    args = parser.parse_args()

    df = pd.read_csv(f'{args.mode}.csv')
    print(df.head())

    patch_h_size, patch_w_size= args.patch_size
    h_resize, w_reszie = args.resize
    h_stride, w_stride = args.stride 

    train_all_input_files = df['LR']
    print(len(train_all_input_files))
    train_input_files = train_all_input_files.to_numpy()
    cut_img(train_input_files, f'./data/{args.mode}/lr', patch_h_size // args.scale, patch_w_size // args.scale, h_resize // args.scale, w_reszie // args.scale, h_stride // args.scale, w_stride // args.scale)

    if args.mode == 'train':
        train_all_label_files = df['HR']
        train_label_files = train_all_label_files.to_numpy()
        cut_img(train_label_files, f'./data/{args.mode}/hr', patch_h_size, patch_w_size, h_resize, w_reszie, h_stride, w_stride)

    print('END')
