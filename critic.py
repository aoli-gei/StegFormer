import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import math
import cv2


def calculate_ssim_skimage(img1, img2):
    """
        计算 ssim
        img1:Tensor
        img2:Tensor
    """
    img_1 = np.array(img1).astype(np.float64)
    img_2 = np.array(img2).astype(np.float64)
    ssim_score=[]
    for (i,j) in zip(img_1,img_2):
        ssim_score.append(structural_similarity(i, j, channel_axis=0,data_range=1))
    return np.mean(ssim_score)

# 使用 skimage 计算 PSNR
def calculate_psnr_skimage(img1, img2):
    """
        calculate psnr in Y channel.
        img1: Tensor
        img2: Tensor
    """
    img_1 = (np.array(img1).astype(np.float64)*255).astype(np.float64)
    img_2 = (np.array(img2).astype(np.float64)*255).astype(np.float64)
    img1_y = rgb2ycbcr(img_1.transpose(0,2,3,1))
    img2_y = rgb2ycbcr(img_2.transpose(0,2,3,1))
    return peak_signal_noise_ratio(img1_y, img2_y,data_range=255)


def calculate_mse(img1, img2):
    test1 = np.array(img1).astype(np.float64)*255
    test2 = np.array(img2).astype(np.float64)*255
    mse = np.mean((test1-test2)**2)
    return mse


def calculate_rmse(img1, img2):
    test1 = np.array(img1).astype(np.float64)*255
    test2 = np.array(img2).astype(np.float64)*255
    rmse = np.sqrt(np.mean((test1-test2)**2))
    return rmse


def calculate_mae(img1, img2):
    test1 = np.array(img1).astype(np.float64)*255
    test2 = np.array(img2).astype(np.float64)*255
    mae = np.mean(np.abs(test1-test2))
    return mae


def calculate_psnr(img1, img2):
    img_1 = np.array(img1).astype(np.float64)*255
    img_2 = np.array(img2).astype(np.float64)*255
    mse = np.mean((img_1 - img_2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# HiNet 指标计算函数


def calculate_psnr_y(img1, img2):
    img_1 = (np.array(img1).astype(np.float64)*255).astype(np.float64)
    img_2 = (np.array(img2).astype(np.float64)*255).astype(np.float64)
    img1_y = rgb2ycbcr(img_1.transpose(1,2,0))
    img2_y = rgb2ycbcr(img_2.transpose(1,2,0))
    mse = np.mean((img1_y - img2_y)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img_1 = np.array(img1).astype(np.float64)
    img_2 = np.array(img2).astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img_1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img_2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img_1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img_2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img_1 * img_2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1_in=np.transpose(img1, (1, 2, 0))
    img2_in=np.transpose(img2, (1, 2, 0))
    if not img1_in.shape == img2_in.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1_in.ndim == 2:
        return ssim(img1_in, img2_in)
    elif img1_in.ndim == 3:
        # 多通道
        if img1_in.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1_in, img2_in))
            return np.array(ssims).mean()
        # 单通道
        elif img1_in.shape[2] == 1:
            return ssim(np.squeeze(img1_in), np.squeeze(img2_in))
    else:
        raise ValueError('Wrong input image dimensions.')


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
