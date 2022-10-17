import torch


def cal_psnr(img1, img2, pixel_max=255.):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(pixel_max / torch.sqrt(mse))
