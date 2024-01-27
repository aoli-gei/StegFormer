import torch
import torch.nn as nn
import torch.optim
import torchvision
import math
import numpy as np
from critic import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
from thop import profile
import torch.nn.functional as F
import os
import timm
import timm.scheduler
from model import StegFormer
from datasets import *
from einops import rearrange
import config
args = config.Args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型初始化
if args.use_model == 'StegFormer-S':
    encoder = StegFormer(img_resolution=args.image_size_train, input_dim=(args.num_secret+1)*3, cnn_emb_dim=8, output_dim=3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret*3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
if args.use_model == 'StegFormer-B':
    encoder = StegFormer(img_resolution=args.image_size_train, input_dim=(args.num_secret+1)*3, cnn_emb_dim=16, output_dim=3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=16, output_dim=args.num_secret*3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act, depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
if args.use_model == 'StegFormer-L':
    encoder = StegFormer(img_resolution=args.image_size_train, input_dim=(args.num_secret+1)*3, cnn_emb_dim=32, output_dim=3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(img_resolution=args.image_size_train, input_dim=3, cnn_emb_dim=32, output_dim=args.num_secret*3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])

# 加载模型
save_path = args.path+'/checkpoint/'+args.model_name
model_path = f'{save_path}/{args.model_name}.pt'
state_dicts = torch.load(model_path)
encoder.load_state_dict(state_dicts['encoder'], strict=False)
decoder.load_state_dict(state_dicts['decoder'], strict=False)

encoder.to(args.device)
decoder.to(args.device)


# 计算模型参数量
with torch.no_grad():
    test_encoder_input = torch.randn(1, (args.num_secret+1)*3, 256, 256).to(args.device)
    test_decoder_input = torch.randn(1, 3, 256, 256).to(args.device)
    encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,))

    decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,))
    print("thop result:encoder FLOPs="+str(encoder_mac*2)+",encoder params="+str(encoder_params))
    print("thop result:decoder FLOPs="+str(decoder_mac*2)+",decoder params="+str(decoder_params))
i = 0   # 为每一张图编号
# 评价指标
psnr_secret = []
psnr_cover = []
psnr_secret_y = []
psnr_seret1=[]
psnr_seret2=[]
psnr_seret3=[]
psnr_seret4=[]

ssim_seret1=[]
ssim_seret2=[]
ssim_seret3=[]
ssim_seret4=[]

psnr_cover_y = []
ssim_secret = []
ssim_cover = []
mse_cover = []
mse_secret = []
rmse_cover = []
rmse_secret = []
mae_cover = []
mae_secret = []

# without clamp
for j in range(1):
    # test 1,000 images
    with torch.no_grad():
        # val
        encoder.eval()
        decoder.eval()

        # 在验证集上测试
        for i_batch, img in enumerate(COCO_test_multi_loader):
            img = img.to(args.device)
            cover = img[0:1, :, :, :]
            secret = img[1:, :, :, :]
            secret_cat = rearrange(secret, '(b n) c h w -> b (n c) h w', n=args.num_secret)  # 通道级联

            # encode
            msg = torch.cat([cover, secret_cat], 1)
            encode_img = encoder(msg)  # 添加残差连接
            encode_img_c = torch.clamp(encode_img, 0, 1)

            # decode
            decode_img = decoder(encode_img_c)
            decode_img = rearrange(decode_img, 'b (n c) h w -> (b n) c h w', n=args.num_secret)

            # 限制为图像表示
            decode_img = decode_img.clamp(0, 1)
            encode_img = encode_img.clamp(0, 1)

            cover_dif=(cover-encode_img)*10
            secret_dif=(secret-decode_img)*10

            # 计算各种指标
            # 拷贝进内存以方便计算
            cover = cover.cpu()
            secret = secret.cpu()
            encode_img = encode_img.cpu()
            decode_img = decode_img.cpu()

            # 移除第一个通道
            decode_img1 = decode_img[:1, :, :, :]
            decode_img2 = decode_img[:2, :, :, :]
            decode_img3 = decode_img[:3, :, :, :]
            decode_img4 = decode_img[:4, :, :, :]

            # 计算 Y 通道 PSNR
            psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
            psnry_decode_temp1 = calculate_psnr_skimage(secret[:1, :, :, :], decode_img1)
            psnry_decode_temp2 = calculate_psnr_skimage(secret[:2, :, :, :], decode_img2)
            psnry_decode_temp3 = calculate_psnr_skimage(secret[:3,:,:,:], decode_img3)
            psnry_decode_temp4 = calculate_psnr_skimage(secret[:4,:,:,:], decode_img4)
            psnry_decode_temp = (psnry_decode_temp1+psnry_decode_temp2+psnry_decode_temp3+psnry_decode_temp4)/4.0
            psnr_seret1.append(psnry_decode_temp1)
            psnr_seret2.append(psnry_decode_temp2)
            psnr_seret3.append(psnry_decode_temp3)
            psnr_seret4.append(psnry_decode_temp4)
            psnr_cover_y.append(psnry_encode_temp)
            psnr_secret_y.append(psnry_decode_temp)

            # 计算 ssim
            ssim_encode_temp = calculate_ssim_skimage(cover, encode_img)
            ssim_decode_temp1 = calculate_ssim_skimage(secret[:1, :, :, :], decode_img1)
            ssim_decode_temp2 = calculate_ssim_skimage(secret[:2, :, :, :], decode_img2)
            ssim_decode_temp3 = calculate_ssim_skimage(secret[:3,:,:,:], decode_img3)
            ssim_decode_temp4 = calculate_ssim_skimage(secret[:4,:,:,:], decode_img4)
            ssim_decode_temp = (ssim_decode_temp1+ssim_decode_temp2+ssim_decode_temp3+ssim_decode_temp4)/4.0
            ssim_seret1.append(ssim_decode_temp1)
            ssim_seret2.append(ssim_decode_temp2)
            ssim_seret3.append(ssim_decode_temp3)
            ssim_seret4.append(ssim_decode_temp4)
            ssim_cover.append(ssim_encode_temp)
            ssim_secret.append(ssim_decode_temp)

            i += 1    # 下一张图像
            print("img "+str(i)+" :")
            print("PSNR_Y_cover:" + str(np.mean(psnr_cover_y)) + " PSNR_Y_secret:" + str(np.mean(psnr_secret_y)))
            print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))

print("clamp total result:")
print("PSNR_Y_cover:" + str(np.mean(psnr_cover_y)) + " PSNR_Y_secret:" + str(np.mean(psnr_secret_y)))
print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
print(f'secret1: {np.mean(psnr_seret1)} {np.mean(ssim_seret1)} secret2: {np.mean(psnr_seret2)} {np.mean(ssim_seret2)} secret3: {np.mean(psnr_seret3)} {np.mean(ssim_seret3)} secret4: {np.mean(psnr_seret4)} {np.mean(ssim_seret4)}')
