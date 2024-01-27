import torch
import torch.nn as nn
import torch.optim
import numpy as np
from critic import *
from thop import profile
import os
from model import StegFormer
from datasets import DIV2K_test_cover_loader, DIV2K_test_secret_loader
import time


# 以类的方式定义参数
class Args:
    def __init__(self) -> None:
        self.batch_size = 8
        self.image_size = 256
        self.patch_size = 16
        self.lr = 1e-3
        self.epochs = 2000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.val_freq = 50
        self.save_freq = 200
        self.train_next = 0
        self.use_model = 'StegFormer-S'
        self.input_dim = 3
        self.num_secret = 1
        self.norm_train = True
        self.output_act=None


args = Args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 多卡训练
USE_MULTI_GPU = False

# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型初始化
if args.use_model == 'StegFormer-S':
    encoder = StegFormer(1024, img_dim=(args.num_secret+1)*3, cnn_emb_dim=8, output_dim=3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act,depth_tr=[2,2,6,2,2,6,2,2])
    decoder = StegFormer(1024, img_dim=3, cnn_emb_dim=6, output_dim=args.num_secret*3, drop_key=False, patch_size=1, window_size=8, output_act=None, depth_tr=[2,2,6,2,2,6,2,2])
if args.use_model == 'StegFormer-B':
    encoder = StegFormer(1024, img_dim=(args.num_secret+1)*3, cnn_emb_dim=16, output_dim=3)
    decoder = StegFormer(1024, img_dim=3, cnn_emb_dim=16, output_dim=args.num_secret*3)
if args.use_model == 'StegFormer-L':
    encoder = StegFormer(1024, img_dim=(args.num_secret+1)*3, cnn_emb_dim=32, output_dim=3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(1024, img_dim=3, cnn_emb_dim=32, output_dim=args.num_secret*3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
encoder.cuda()
decoder.cuda()


# 加载模型
model_path = '/home/whq135/code/stegv1/model_uformer/StegFormer_HELD3.pt'
state_dicts = torch.load(model_path)
encoder.load_state_dict(state_dicts['encoder'], strict=False)
decoder.load_state_dict(state_dicts['decoder'], strict=False)

# 数据并行
if USE_MULTI_GPU:
    encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
    decoder = torch.nn.DataParallel(decoder, device_ids=device_ids)
encoder.to(device)
decoder.to(device)

# 计算模型参数量
with torch.no_grad():
    test_encoder_input = torch.randn(1, 6, 1024, 1024).to(device)
    test_decoder_input = torch.randn(1, 3, 1024, 1024).to(device)
    encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,))

    decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,))
    print("thop result:encoder FLOPs="+str(encoder_mac*2)+",encoder params="+str(encoder_params))
    print("thop result:decoder FLOPs="+str(decoder_mac*2)+",decoder params="+str(decoder_params))

i = 0   # 为每一张图编号
# 评价指标


# without clamp
with torch.no_grad():
    # val
    encoder.eval()
    decoder.eval()

    psnr_secret = []
    psnr_cover = []
    psnr_secret_y = []
    psnr_cover_y = []
    ssim_secret = []
    ssim_cover = []
    mse_cover = []
    mse_secret = []
    rmse_cover = []
    rmse_secret = []
    mae_cover = []
    mae_secret = []

    # 在 DIV2K
    i = 0
    for j in range(1):  # 需要的轮次
        break
        for (cover, secret) in zip(DIV2K_test_cover_loader, DIV2K_test_secret_loader):
            cover = cover.to(device)
            secret = secret.to(device)

            # encode
            msg = torch.cat([cover, secret], 1)
            encode_img = encoder(msg)  # 添加残差连接

            # decode
            decode_img = decoder(encode_img)

            # 限制为图像表示
            decode_img = decode_img.clamp(0, 1)
            encode_img = encode_img.clamp(0, 1)

            # 计算各种指标
            # 拷贝进内存以方便计算
            cover = cover.cpu()
            secret = secret.cpu()
            encode_img = encode_img.cpu()
            decode_img = decode_img.cpu()

            # # 计算 RGB PSNR
            # psnr_encode_temp = calculate_psnr(cover, encode_img)
            # psnr_decode_temp = calculate_psnr(secret, decode_img)
            # psnr_cover.append(psnr_encode_temp)
            # psnr_secret.append(psnr_decode_temp)

            # 计算 Y 通道 PSNR
            psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
            psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
            psnr_cover_y.append(psnry_encode_temp)
            psnr_secret_y.append(psnry_decode_temp)

            # 计算 SSIM
            ssim_cover_temp=calculate_ssim_skimage(cover,encode_img)
            ssim_secret_temp=calculate_ssim_skimage(secret,decode_img)
            ssim_cover.append(ssim_cover_temp)
            ssim_secret.append(ssim_secret_temp)

            # # 计算 MSE
            # mse_cover_temp = calculate_mse(cover, encode_img)
            # mse_secret_temp = calculate_mse(secret, decode_img)
            # mse_cover.append(mse_cover_temp)
            # mse_secret.append(mse_secret_temp)

            # 计算 RMSE
            rmse_cover_temp = calculate_rmse(cover, encode_img)
            rmse_secret_temp = calculate_rmse(secret, decode_img)
            rmse_cover.append(rmse_cover_temp)
            rmse_secret.append(rmse_secret_temp)

            # 计算 MAE
            mae_cover_temp = calculate_mae(cover, encode_img)
            mae_secret_temp = calculate_mae(secret, decode_img)
            mae_cover.append(mae_cover_temp)
            mae_secret.append(mae_secret_temp)

            # 保存图像
            # torchvision.utils.save_image(cover, '/home/whq135/code/stegv1/image/wo_clamp/cover/' + '%.5d.png' % i)
            # torchvision.utils.save_image(secret, '/home/whq135/code/stegv1/image/wo_clamp/secret/' + '%.5d.png' % i)
            # torchvision.utils.save_image(encode_img, '/home/whq135/code/stegv1/image/wo_clamp/stego/' + '%.5d.png' % i)
            # torchvision.utils.save_image(decode_img, '/home/whq135/code/stegv1/image/wo_clamp/secret-rev/' + '%.5d.png' % i)
            i += 1    # 下一张图像
            print(f'item {i} : cover_psnr: {psnry_encode_temp} cover_ssim: {ssim_cover_temp} cover_mae: {mae_cover_temp} cover_rmse: {rmse_cover_temp}\n \
                  secret_psnr: {psnry_decode_temp} secret_ssim: {ssim_secret_temp} secret_mae: {mae_secret_temp} secret_rmse: {rmse_secret_temp}')
    cover_div2k_psnr = np.mean(psnr_cover_y)
    cover_div2k_ssim = np.mean(ssim_cover)
    cover_div2k_mae = np.mean(mae_cover)
    cover_div2k_rmse = np.mean(rmse_cover)
    secret_div2k_psnr = np.mean(psnr_secret_y)
    secret_div2k_ssim = np.mean(ssim_secret)
    secret_div2k_mae = np.mean(mae_secret)
    secret_div2k_rmse = np.mean(rmse_secret)
    print('DIV2K:')
    print(f'cover:\n \
            psnr: {cover_div2k_psnr}; ssim: {cover_div2k_ssim}; mae: {cover_div2k_mae}; rmse: {cover_div2k_rmse};\n \
            secret:\n \
            psnr: {secret_div2k_psnr}; ssim: {secret_div2k_ssim}; mae: {secret_div2k_mae}; rmse: {secret_div2k_rmse};\n')

    # # 重载分辨率大小为 256
    # encoder = StegFormer(256, img_dim=(args.num_secret+1)*3, cnn_emb_dim=8, output_dim=3)
    # decoder = StegFormer(256, img_dim=3, cnn_emb_dim=8, output_dim=args.num_secret*3)

    # # 加载模型
    # model_path = '/home/whq135/code/stegv1/model_uformer/StegFormer_woClamp_nuew.pt'
    # state_dicts = torch.load(model_path)
    # encoder.load_state_dict(state_dicts['encoder'], strict=False)
    # decoder.load_state_dict(state_dicts['decoder'], strict=False)
    # encoder.to(device)
    # decoder.to(device)

    # psnr_secret = []
    # psnr_cover = []
    # psnr_secret_y = []
    # psnr_cover_y = []
    # ssim_secret = []
    # ssim_cover = []
    # mse_cover = []
    # mse_secret = []
    # rmse_cover = []
    # rmse_secret = []
    # mae_cover = []
    # mae_secret = []

    # # 在 COCO
    # i = 0
    # for j in range(1):  # 需要的轮次
    #     for (cover, secret) in zip(COCO_test_cover_loader, COCO_test_secret_loader):
    #         if i == 1000:
    #             break
    #         else:
    #             cover = cover.to(device)
    #             secret = secret.to(device)

    #             # encode
    #             msg = torch.cat([cover, secret], 1)
    #             encode_img = encoder(msg)  # 添加残差连接

    #             # decode
    #             decode_img = decoder(encode_img)

    #             # 限制为图像表示
    #             decode_img = decode_img.clamp(0, 1)
    #             encode_img = encode_img.clamp(0, 1)

    #             # 计算各种指标
    #             # 拷贝进内存以方便计算
    #             cover = cover.cpu()
    #             secret = secret.cpu()
    #             encode_img = encode_img.cpu()
    #             decode_img = decode_img.cpu()

    #             # # 计算 RGB PSNR
    #             # psnr_encode_temp = calculate_psnr(cover, encode_img)
    #             # psnr_decode_temp = calculate_psnr(secret, decode_img)
    #             # psnr_cover.append(psnr_encode_temp)
    #             # psnr_secret.append(psnr_decode_temp)

    #             # 计算 Y 通道 PSNR
    #             psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
    #             psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
    #             psnr_cover_y.append(psnry_encode_temp)
    #             psnr_secret_y.append(psnry_decode_temp)

    #             # 计算 SSIM
    #             ssim_cover_temp=calculate_ssim_skimage(cover,encode_img)
    #             ssim_secret_temp=calculate_ssim_skimage(secret,decode_img)
    #             ssim_cover.append(ssim_cover_temp)
    #             ssim_secret.append(ssim_secret_temp)

    #             # # 计算 MSE
    #             # mse_cover_temp = calculate_mse(cover, encode_img)
    #             # mse_secret_temp = calculate_mse(secret, decode_img)
    #             # mse_cover.append(mse_cover_temp)
    #             # mse_secret.append(mse_secret_temp)

    #             # 计算 RMSE
    #             rmse_cover_temp = calculate_rmse(cover, encode_img)
    #             rmse_secret_temp = calculate_rmse(secret, decode_img)
    #             rmse_cover.append(rmse_cover_temp)
    #             rmse_secret.append(rmse_secret_temp)

    #             # 计算 MAE
    #             mae_cover_temp = calculate_mae(cover, encode_img)
    #             mae_secret_temp = calculate_mae(secret, decode_img)
    #             mae_cover.append(mae_cover_temp)
    #             mae_secret.append(mae_secret_temp)

    #             # # 保存图像
    #             # torchvision.utils.save_image(cover, '/home/whq135/code/stegv1/image/wo_clamp/cover/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(secret, '/home/whq135/code/stegv1/image/wo_clamp/secret/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(encode_img, '/home/whq135/code/stegv1/image/wo_clamp/stego/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(decode_img, '/home/whq135/code/stegv1/image/wo_clamp/secret-rev/' + '%.5d.png' % i)
    #             i += 1    # 下一张图像
    #             # print(f'item {i} : cover_psnr: {psnry_encode_temp} cover_ssim: {ssim_cover_temp} cover_mae: {mae_cover_temp} cover_rmse: {rmse_cover_temp}\n \
    #             #     secret_psnr: {psnry_decode_temp} secret_ssim: {ssim_cover_temp} secret_mae: {mae_secret_temp} secret_rmse: {rmse_secret_temp}')
    # cover_coco_psnr = np.mean(psnr_cover_y)
    # cover_coco_ssim = np.mean(ssim_cover)
    # cover_coco_mae = np.mean(mae_cover)
    # cover_coco_rmse = np.mean(rmse_cover)
    # secret_coco_psnr = np.mean(psnr_secret_y)
    # secret_coco_ssim = np.mean(ssim_secret)
    # secret_coco_mae = np.mean(mae_secret)
    # secret_coco_rmse = np.mean(rmse_secret)
    # print('COCO')
    # print(f'cover:\n \
    #         psnr: {cover_coco_psnr}; ssim: {cover_coco_ssim}; mae: {cover_coco_mae}; rmse: {cover_coco_rmse};\n \
    #         secret:\n \
    #         psnr: {secret_coco_psnr}; ssim: {secret_coco_ssim}; mae: {secret_coco_mae}; rmse: {secret_coco_rmse};\n')

    # psnr_secret = []
    # psnr_cover = []
    # psnr_secret_y = []
    # psnr_cover_y = []
    # ssim_secret = []
    # ssim_cover = []
    # mse_cover = []
    # mse_secret = []
    # rmse_cover = []
    # rmse_secret = []
    # mae_cover = []
    # mae_secret = []

    # # 在 ImageNet
    # i = 0
    # for j in range(1):  # 需要的轮次
    #     for (cover, secret) in zip(ImageNet_test_cover_loader, ImageNet_test_secret_loader):
    #         if i == 1000:
    #             break
    #         else:
    #             cover = cover.to(device)
    #             secret = secret.to(device)

    #             # encode
    #             msg = torch.cat([cover, secret], 1)
    #             encode_img = encoder(msg)  # 添加残差连接

    #             # decode
    #             decode_img = decoder(encode_img)

    #             # 限制为图像表示
    #             decode_img = decode_img.clamp(0, 1)
    #             encode_img = encode_img.clamp(0, 1)

    #             # 计算各种指标
    #             # 拷贝进内存以方便计算
    #             cover = cover.cpu()
    #             secret = secret.cpu()
    #             encode_img = encode_img.cpu()
    #             decode_img = decode_img.cpu()

    #             # # 计算 RGB PSNR
    #             # psnr_encode_temp = calculate_psnr(cover, encode_img)
    #             # psnr_decode_temp = calculate_psnr(secret, decode_img)
    #             # psnr_cover.append(psnr_encode_temp)
    #             # psnr_secret.append(psnr_decode_temp)

    #             # 计算 Y 通道 PSNR
    #             psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
    #             psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
    #             psnr_cover_y.append(psnry_encode_temp)
    #             psnr_secret_y.append(psnry_decode_temp)

    #             # 计算 SSIM
    #             ssim_cover_temp=calculate_ssim_skimage(cover,encode_img)
    #             ssim_secret_temp=calculate_ssim_skimage(secret,decode_img)
    #             ssim_cover.append(ssim_cover_temp)
    #             ssim_secret.append(ssim_secret_temp)

    #             # # 计算 MSE
    #             # mse_cover_temp = calculate_mse(cover, encode_img)
    #             # mse_secret_temp = calculate_mse(secret, decode_img)
    #             # mse_cover.append(mse_cover_temp)
    #             # mse_secret.append(mse_secret_temp)

    #             # 计算 RMSE
    #             rmse_cover_temp = calculate_rmse(cover, encode_img)
    #             rmse_secret_temp = calculate_rmse(secret, decode_img)
    #             rmse_cover.append(rmse_cover_temp)
    #             rmse_secret.append(rmse_secret_temp)

    #             # 计算 MAE
    #             mae_cover_temp = calculate_mae(cover, encode_img)
    #             mae_secret_temp = calculate_mae(secret, decode_img)
    #             mae_cover.append(mae_cover_temp)
    #             mae_secret.append(mae_secret_temp)

    #             # # 保存图像
    #             # torchvision.utils.save_image(cover, '/home/whq135/code/stegv1/image/wo_clamp/cover/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(secret, '/home/whq135/code/stegv1/image/wo_clamp/secret/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(encode_img, '/home/whq135/code/stegv1/image/wo_clamp/stego/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(decode_img, '/home/whq135/code/stegv1/image/wo_clamp/secret-rev/' + '%.5d.png' % i)
    #             i += 1    # 下一张图像
    #             # print(f'item {i} : cover_psnr: {psnry_encode_temp} cover_ssim: {ssim_cover_temp} cover_mae: {mae_cover_temp} cover_rmse: {rmse_cover_temp}\n \
    #             #     secret_psnr: {psnry_decode_temp} secret_ssim: {ssim_cover_temp} secret_mae: {mae_secret_temp} secret_rmse: {rmse_secret_temp}')
    # cover_imagenet_psnr = np.mean(psnr_cover_y)
    # cover_imagenet_ssim = np.mean(ssim_cover)
    # cover_imagenet_mae = np.mean(mae_cover)
    # cover_imagenet_rmse = np.mean(rmse_cover)
    # secret_imagenet_psnr = np.mean(psnr_secret_y)
    # secret_imagenet_ssim = np.mean(ssim_secret)
    # secret_imagenet_mae = np.mean(mae_secret)
    # secret_imagenet_rmse = np.mean(rmse_secret)
    # print('imagenet')
    # print(f'cover:\n \
    #         psnr: {cover_imagenet_psnr}; ssim: {cover_imagenet_ssim}; mae: {cover_imagenet_mae}; rmse: {cover_imagenet_rmse};\n \
    #         secret:\n \
    #         psnr: {secret_imagenet_psnr}; ssim: {secret_imagenet_ssim}; mae: {secret_imagenet_mae}; rmse: {secret_imagenet_rmse};\n')


# clamp
print('clamp')
with torch.no_grad():
    # val
    encoder = StegFormer(1024, img_dim=(args.num_secret+1)*3, cnn_emb_dim=12, output_dim=3,
                         drop_key=False, patch_size=2, window_size=8, output_act=args.output_act,depth_tr=[2,2,2,2,2,2,2,2])
    decoder = StegFormer(1024, img_dim=3, cnn_emb_dim=6, output_dim=args.num_secret*3, drop_key=False, patch_size=1, window_size=8, output_act=None, depth_tr=[2,2,6,2,2,6,2,2])
    # 加载模型
    model_path = '/home/whq135/code/stegv1/model_uformer/StegFormer_HELD13.pt'
    state_dicts = torch.load(model_path)
    encoder.load_state_dict(state_dicts['encoder'], strict=False)
    decoder.load_state_dict(state_dicts['decoder'], strict=False)
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    psnr_secret = []
    psnr_cover = []
    psnr_secret_y = []
    psnr_cover_y = []
    ssim_secret = []
    ssim_cover = []
    mse_cover = []
    mse_secret = []
    rmse_cover = []
    rmse_secret = []
    mae_cover = []
    mae_secret = []
    encode_times=[]
    decode_times=[]

    # 在 DIV2K
    i = 0
    for j in range(1):  # 需要的轮次
        for (cover, secret) in zip(DIV2K_test_cover_loader, DIV2K_test_secret_loader):
            cover = cover.to(device)
            secret = secret.to(device)

            # encode
            msg = torch.cat([cover, secret], 1)
            start_time=time.perf_counter()
            encode_img = encoder(msg)  # 添加残差连接
            end_time=time.perf_counter()
            encode_time=end_time-start_time
            encode_times.append(encode_time)
            encode_img = torch.clamp(encode_img,0,1)

            # decode
            start_time=time.perf_counter()
            decode_img = decoder(encode_img)
            end_time=time.perf_counter()
            decode_time=end_time-start_time
            decode_times.append(decode_time)

            # 限制为图像表示
            decode_img = decode_img.clamp(0, 1)
            encode_img = encode_img.clamp(0, 1)
            print(f'{i} encode time:{encode_time}; decode time: {decode_time}')

            # 计算各种指标
            # 拷贝进内存以方便计算
            cover = cover.cpu()
            secret = secret.cpu()
            encode_img = encode_img.cpu()
            decode_img = decode_img.cpu()

            # # 计算 RGB PSNR
            # psnr_encode_temp = calculate_psnr(cover, encode_img)
            # psnr_decode_temp = calculate_psnr(secret, decode_img)
            # psnr_cover.append(psnr_encode_temp)
            # psnr_secret.append(psnr_decode_temp)

            # 计算 Y 通道 PSNR
            psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
            psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
            psnr_cover_y.append(psnry_encode_temp)
            psnr_secret_y.append(psnry_decode_temp)

            # 计算 SSIM
            ssim_cover_temp=calculate_ssim_skimage(cover,encode_img)
            ssim_secret_temp=calculate_ssim_skimage(secret,decode_img)
            ssim_cover.append(ssim_cover_temp)
            ssim_secret.append(ssim_secret_temp)

            # # 计算 MSE
            # mse_cover_temp = calculate_mse(cover, encode_img)
            # mse_secret_temp = calculate_mse(secret, decode_img)
            # mse_cover.append(mse_cover_temp)
            # mse_secret.append(mse_secret_temp)

            # 计算 RMSE
            rmse_cover_temp = calculate_rmse(cover, encode_img)
            rmse_secret_temp = calculate_rmse(secret, decode_img)
            rmse_cover.append(rmse_cover_temp)
            rmse_secret.append(rmse_secret_temp)

            # 计算 MAE
            mae_cover_temp = calculate_mae(cover, encode_img)
            mae_secret_temp = calculate_mae(secret, decode_img)
            mae_cover.append(mae_cover_temp)
            mae_secret.append(mae_secret_temp)

            # # 保存图像
            # torchvision.utils.save_image(cover, '/home/whq135/code/stegv1/image/wo_clamp/cover/' + '%.5d.png' % i)
            # torchvision.utils.save_image(secret, '/home/whq135/code/stegv1/image/wo_clamp/secret/' + '%.5d.png' % i)
            # torchvision.utils.save_image(encode_img, '/home/whq135/code/stegv1/image/wo_clamp/stego/' + '%.5d.png' % i)
            # torchvision.utils.save_image(decode_img, '/home/whq135/code/stegv1/image/wo_clamp/secret-rev/' + '%.5d.png' % i)
            i += 1    # 下一张图像
            print(f'item {i} : cover_psnr: {psnry_encode_temp} cover_ssim: {ssim_cover_temp} cover_mae: {mae_cover_temp} cover_rmse: {rmse_cover_temp}\n \
                  secret_psnr: {psnry_decode_temp} secret_ssim: {ssim_secret_temp} secret_mae: {mae_secret_temp} secret_rmse: {rmse_secret_temp}')
    cover_div2k_psnr = np.mean(psnr_cover_y)
    cover_div2k_ssim = np.mean(ssim_cover)
    cover_div2k_mae = np.mean(mae_cover)
    cover_div2k_rmse = np.mean(rmse_cover)
    secret_div2k_psnr = np.mean(psnr_secret_y)
    secret_div2k_ssim = np.mean(ssim_secret)
    secret_div2k_mae = np.mean(mae_secret)
    secret_div2k_rmse = np.mean(rmse_secret)
    print('DIV2K:')
    print(f'cover:\n \
            psnr: {cover_div2k_psnr}; ssim: {cover_div2k_ssim}; mae: {cover_div2k_mae}; rmse: {cover_div2k_rmse};\n \
            secret:\n \
            psnr: {secret_div2k_psnr}; ssim: {secret_div2k_ssim}; mae: {secret_div2k_mae}; rmse: {secret_div2k_rmse};\n')
    print(f'encode_time:{np.mean(encode_times)}')
    print(f'decode_time:{np.mean(decode_times)}')

    # # 重载分辨率大小为 256
    # encoder = StegFormer(256, img_dim=(args.num_secret+1)*3, cnn_emb_dim=8, output_dim=3)
    # decoder = StegFormer(256, img_dim=3, cnn_emb_dim=8, output_dim=args.num_secret*3)

    # # 加载模型
    # model_path = '/home/whq135/code/stegv1/model_uformer/StegFormer_Clamp_nuew.pt'
    # state_dicts = torch.load(model_path)
    # encoder.load_state_dict(state_dicts['encoder'], strict=False)
    # decoder.load_state_dict(state_dicts['decoder'], strict=False)
    # encoder.to(device)
    # decoder.to(device)

    # psnr_secret = []
    # psnr_cover = []
    # psnr_secret_y = []
    # psnr_cover_y = []
    # ssim_secret = []
    # ssim_cover = []
    # mse_cover = []
    # mse_secret = []
    # rmse_cover = []
    # rmse_secret = []
    # mae_cover = []
    # mae_secret = []

    # # 在 COCO
    # i = 0
    # for j in range(1):  # 需要的轮次
    #     for (cover, secret) in zip(COCO_test_cover_loader, COCO_test_secret_loader):
    #         if i == 1000:
    #             break
    #         else:
    #             cover = cover.to(device)
    #             secret = secret.to(device)

    #             # encode
    #             msg = torch.cat([cover, secret], 1)
    #             encode_img = encoder(msg)  # 添加残差连接
    #             encode_img = torch.clamp(encode_img,0,1)

    #             # decode
    #             decode_img = decoder(encode_img)

    #             # 限制为图像表示
    #             decode_img = decode_img.clamp(0, 1)
    #             encode_img = encode_img.clamp(0, 1)

    #             # 计算各种指标
    #             # 拷贝进内存以方便计算
    #             cover = cover.cpu()
    #             secret = secret.cpu()
    #             encode_img = encode_img.cpu()
    #             decode_img = decode_img.cpu()

    #             # # 计算 RGB PSNR
    #             # psnr_encode_temp = calculate_psnr(cover, encode_img)
    #             # psnr_decode_temp = calculate_psnr(secret, decode_img)
    #             # psnr_cover.append(psnr_encode_temp)
    #             # psnr_secret.append(psnr_decode_temp)

    #             # 计算 Y 通道 PSNR
    #             psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
    #             psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
    #             psnr_cover_y.append(psnry_encode_temp)
    #             psnr_secret_y.append(psnry_decode_temp)

    #             # 计算 SSIM
    #             ssim_cover_temp=calculate_ssim_skimage(cover,encode_img)
    #             ssim_secret_temp=calculate_ssim_skimage(secret,decode_img)
    #             ssim_cover.append(ssim_cover_temp)
    #             ssim_secret.append(ssim_secret_temp)

    #             # # 计算 MSE
    #             # mse_cover_temp = calculate_mse(cover, encode_img)
    #             # mse_secret_temp = calculate_mse(secret, decode_img)
    #             # mse_cover.append(mse_cover_temp)
    #             # mse_secret.append(mse_secret_temp)

    #             # 计算 RMSE
    #             rmse_cover_temp = calculate_rmse(cover, encode_img)
    #             rmse_secret_temp = calculate_rmse(secret, decode_img)
    #             rmse_cover.append(rmse_cover_temp)
    #             rmse_secret.append(rmse_secret_temp)

    #             # 计算 MAE
    #             mae_cover_temp = calculate_mae(cover, encode_img)
    #             mae_secret_temp = calculate_mae(secret, decode_img)
    #             mae_cover.append(mae_cover_temp)
    #             mae_secret.append(mae_secret_temp)

    #             # # 保存图像
    #             # torchvision.utils.save_image(cover, '/home/whq135/code/stegv1/image/wo_clamp/cover/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(secret, '/home/whq135/code/stegv1/image/wo_clamp/secret/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(encode_img, '/home/whq135/code/stegv1/image/wo_clamp/stego/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(decode_img, '/home/whq135/code/stegv1/image/wo_clamp/secret-rev/' + '%.5d.png' % i)
    #             i += 1    # 下一张图像
    #             # print(f'item {i} : cover_psnr: {psnry_encode_temp} cover_ssim: {ssim_cover_temp} cover_mae: {mae_cover_temp} cover_rmse: {rmse_cover_temp}\n \
    #             #     secret_psnr: {psnry_decode_temp} secret_ssim: {ssim_cover_temp} secret_mae: {mae_secret_temp} secret_rmse: {rmse_secret_temp}')
    # cover_coco_psnr = np.mean(psnr_cover_y)
    # cover_coco_ssim = np.mean(ssim_cover)
    # cover_coco_mae = np.mean(mae_cover)
    # cover_coco_rmse = np.mean(rmse_cover)
    # secret_coco_psnr = np.mean(psnr_secret_y)
    # secret_coco_ssim = np.mean(ssim_secret)
    # secret_coco_mae = np.mean(mae_secret)
    # secret_coco_rmse = np.mean(rmse_secret)
    # print('COCO')
    # print(f'cover:\n \
    #         psnr: {cover_coco_psnr}; ssim: {cover_coco_ssim}; mae: {cover_coco_mae}; rmse: {cover_coco_rmse};\n \
    #         secret:\n \
    #         psnr: {secret_coco_psnr}; ssim: {secret_coco_ssim}; mae: {secret_coco_mae}; rmse: {secret_coco_rmse};\n')

    # psnr_secret = []
    # psnr_cover = []
    # psnr_secret_y = []
    # psnr_cover_y = []
    # ssim_secret = []
    # ssim_cover = []
    # mse_cover = []
    # mse_secret = []
    # rmse_cover = []
    # rmse_secret = []
    # mae_cover = []
    # mae_secret = []

    # # 在 ImageNet
    # i = 0
    # for j in range(1):  # 需要的轮次
    #     for (cover, secret) in zip(ImageNet_test_cover_loader, ImageNet_test_secret_loader):
    #         if i == 1000:
    #             break
    #         else:
    #             cover = cover.to(device)
    #             secret = secret.to(device)

    #             # encode
    #             msg = torch.cat([cover, secret], 1)
    #             encode_img = encoder(msg)  # 添加残差连接
    #             encode_img = torch.clamp(encode_img,0,1)

    #             # decode
    #             decode_img = decoder(encode_img)

    #             # 限制为图像表示
    #             decode_img = decode_img.clamp(0, 1)
    #             encode_img = encode_img.clamp(0, 1)

    #             # 计算各种指标
    #             # 拷贝进内存以方便计算
    #             cover = cover.cpu()
    #             secret = secret.cpu()
    #             encode_img = encode_img.cpu()
    #             decode_img = decode_img.cpu()

    #             # # 计算 RGB PSNR
    #             # psnr_encode_temp = calculate_psnr(cover, encode_img)
    #             # psnr_decode_temp = calculate_psnr(secret, decode_img)
    #             # psnr_cover.append(psnr_encode_temp)
    #             # psnr_secret.append(psnr_decode_temp)

    #             # 计算 Y 通道 PSNR
    #             psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
    #             psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
    #             psnr_cover_y.append(psnry_encode_temp)
    #             psnr_secret_y.append(psnry_decode_temp)

    #             # 计算 SSIM
    #             ssim_cover_temp=calculate_ssim_skimage(cover,encode_img)
    #             ssim_secret_temp=calculate_ssim_skimage(secret,decode_img)
    #             ssim_cover.append(ssim_cover_temp)
    #             ssim_secret.append(ssim_secret_temp)

    #             # # 计算 MSE
    #             # mse_cover_temp = calculate_mse(cover, encode_img)
    #             # mse_secret_temp = calculate_mse(secret, decode_img)
    #             # mse_cover.append(mse_cover_temp)
    #             # mse_secret.append(mse_secret_temp)

    #             # 计算 RMSE
    #             rmse_cover_temp = calculate_rmse(cover, encode_img)
    #             rmse_secret_temp = calculate_rmse(secret, decode_img)
    #             rmse_cover.append(rmse_cover_temp)
    #             rmse_secret.append(rmse_secret_temp)

    #             # 计算 MAE
    #             mae_cover_temp = calculate_mae(cover, encode_img)
    #             mae_secret_temp = calculate_mae(secret, decode_img)
    #             mae_cover.append(mae_cover_temp)
    #             mae_secret.append(mae_secret_temp)

    #             # # 保存图像
    #             # torchvision.utils.save_image(cover, '/home/whq135/code/stegv1/image/wo_clamp/cover/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(secret, '/home/whq135/code/stegv1/image/wo_clamp/secret/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(encode_img, '/home/whq135/code/stegv1/image/wo_clamp/stego/' + '%.5d.png' % i)
    #             # torchvision.utils.save_image(decode_img, '/home/whq135/code/stegv1/image/wo_clamp/secret-rev/' + '%.5d.png' % i)
    #             i += 1    # 下一张图像
    #             # print(f'item {i} : cover_psnr: {psnry_encode_temp} cover_ssim: {ssim_cover_temp} cover_mae: {mae_cover_temp} cover_rmse: {rmse_cover_temp}\n \
    #             #     secret_psnr: {psnry_decode_temp} secret_ssim: {ssim_cover_temp} secret_mae: {mae_secret_temp} secret_rmse: {rmse_secret_temp}')
    # cover_imagenet_psnr = np.mean(psnr_cover_y)
    # cover_imagenet_ssim = np.mean(ssim_cover)
    # cover_imagenet_mae = np.mean(mae_cover)
    # cover_imagenet_rmse = np.mean(rmse_cover)
    # secret_imagenet_psnr = np.mean(psnr_secret_y)
    # secret_imagenet_ssim = np.mean(ssim_secret)
    # secret_imagenet_mae = np.mean(mae_secret)
    # secret_imagenet_rmse = np.mean(rmse_secret)
    # print('imagenet')
    # print(f'cover:\n \
    #         psnr: {cover_imagenet_psnr}; ssim: {cover_imagenet_ssim}; mae: {cover_imagenet_mae}; rmse: {cover_imagenet_rmse};\n \
    #         secret:\n \
    #         psnr: {secret_imagenet_psnr}; ssim: {secret_imagenet_ssim}; mae: {secret_imagenet_mae}; rmse: {secret_imagenet_rmse};\n')
