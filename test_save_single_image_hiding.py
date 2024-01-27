import torch
import torch.nn as nn
import torch.optim
import torchvision
import numpy as np
from critic import *
from thop import profile
import os
from model import StegFormer
from datasets import *
import config
args = config.Args()

# initialization
if args.use_model == 'StegFormer-S':
    encoder = StegFormer(1024, input_dim=(args.num_secret+1)*3, cnn_emb_dim=8, output_dim=3)
    decoder = StegFormer(1024, input_dim=3, cnn_emb_dim=8, output_dim=args.num_secret*3)
if args.use_model == 'StegFormer-B':
    encoder = StegFormer(1024, input_dim=(args.num_secret+1)*3, cnn_emb_dim=16, output_dim=3,
                         drop_key=False, patch_size=2, window_size=8, output_act=None,depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(1024, input_dim=3, cnn_emb_dim=16, output_dim=args.num_secret*3,
                         drop_key=False, patch_size=2, window_size=8, output_act=None,depth=[1, 1, 1, 1, 2, 1, 1, 1, 1], depth_tr=[2, 2, 2, 2, 2, 2, 2, 2])
if args.use_model == 'StegFormer-L':
    encoder = StegFormer(1024, input_dim=(args.num_secret+1)*3, cnn_emb_dim=32, output_dim=3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])
    decoder = StegFormer(1024, input_dim=3, cnn_emb_dim=32, output_dim=args.num_secret*3, depth=[2, 2, 2, 2, 2, 2, 2, 2, 2])

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
    test_encoder_input = torch.randn(1, 6, 1024, 1024).to(args.device)
    test_decoder_input = torch.randn(1, 3, 1024, 1024).to(args.device)
    encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,))

    decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,))
    print("thop result:encoder FLOPs="+str(encoder_mac*2)+",encoder params="+str(encoder_params))
    print("thop result:decoder FLOPs="+str(decoder_mac*2)+",decoder params="+str(decoder_params))

i = 0   # 为每一张图编号
# 评价指标
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

# without clamp
for j in range(1):
    with torch.no_grad():
        # val
        encoder.eval()
        decoder.eval()

        # 在验证集上测试
        for (cover, secret) in zip(DIV2K_test_cover_loader, DIV2K_test_secret_loader):
            cover = cover.to(args.device)
            secret = secret.to(args.device)

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

            # 计算 Y 通道 PSNR
            psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
            psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
            psnr_cover_y.append(psnry_encode_temp)
            psnr_secret_y.append(psnry_decode_temp)

            # 计算 SSIM
            ssim_encode=calculate_ssim_skimage(cover,encode_img)
            ssim_decode=calculate_ssim_skimage(secret,decode_img)
            ssim_cover.append(ssim_encode)
            ssim_secret.append(ssim_decode)

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
            torchvision.utils.save_image(cover, args.path + '/image/wo_clamp/cover/' + '%.5d.png' % i)
            torchvision.utils.save_image(secret, args.path + '/image/wo_clamp/secret/' + '%.5d.png' % i)
            torchvision.utils.save_image(encode_img, args.path + '/image/wo_clamp/stego/' + '%.5d.png' % i)
            torchvision.utils.save_image(decode_img, args.path + '/image/wo_clamp/secret_rev/' + '%.5d.png' % i)
            i += 1    # 下一张图像
            print("img "+str(i)+" :")
            print("PSNR_Y_cover:" + str(np.mean(psnry_encode_temp)) + " PSNR_Y_secret:" + str(np.mean(psnry_decode_temp)))
            print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
            print("RMSE_cover:" + str(np.mean(rmse_cover_temp)) + " RMSE_secret:" + str(np.mean(rmse_secret_temp)))
            print("MAE_cover:" + str(np.mean(mae_cover_temp)) + " MAE_secret:" + str(np.mean(mae_secret_temp)))

print("wo_clamp total result:")
print("PSNR_Y_cover:" + str(np.mean(psnr_cover_y)) + " PSNR_Y_secret:" + str(np.mean(psnr_secret_y)))
print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
print("MAE_cover:" + str(np.mean(mae_cover)) + " MSE_secret:" + str(np.mean(mae_secret)))
print("RMSE_cover:" + str(np.mean(rmse_cover)) + " RMSE_secret:" + str(np.mean(rmse_secret)))

# 计算 clamp 的指标
i = 0   # 为每一张图编号
# 评价指标
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

# with clamp
for j in range(1):
    # test 1,000 images
    with torch.no_grad():
        # val
        encoder.eval()
        decoder.eval()

        # 在验证集上测试
        for (cover, secret) in zip(DIV2K_test_cover_loader, DIV2K_test_secret_loader):
            cover = cover.to(args.device)
            secret = secret.to(args.device)

            # encode
            msg = torch.cat([cover, secret], 1)
            encode_img = encoder(msg)  # 添加残差连接
            encode_img=torch.clamp(encode_img,0,1)

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

            # 计算 Y 通道 PSNR
            psnry_encode_temp = calculate_psnr_skimage(cover, encode_img)
            psnry_decode_temp = calculate_psnr_skimage(secret, decode_img)
            psnr_cover_y.append(psnry_encode_temp)
            psnr_secret_y.append(psnry_decode_temp)

            # 计算 SSIM
            ssim_encode=calculate_ssim_skimage(cover,encode_img)
            ssim_decode=calculate_ssim_skimage(secret,decode_img)
            ssim_cover.append(ssim_encode)
            ssim_secret.append(ssim_decode)

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
            torchvision.utils.save_image(cover, args.path + '/image/clamp/cover/' + '%.5d.png' % i)
            torchvision.utils.save_image(secret, args.path + '/image/clamp/secret/' + '%.5d.png' % i)
            torchvision.utils.save_image(encode_img, args.path + '/image/clamp/stego/' + '%.5d.png' % i)
            torchvision.utils.save_image(decode_img, args.path + '/image/clamp/secret_rev/' + '%.5d.png' % i)
            i += 1    # 下一张图像
            # print("img "+str(i)+" :")
            # print("PSNR_Y_cover:" + str(np.mean(psnry_encode_temp)) + " PSNR_Y_secret:" + str(np.mean(psnry_decode_temp)))
            # print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
            # print("RMSE_cover:" + str(np.mean(rmse_cover_temp)) + " RMSE_secret:" + str(np.mean(rmse_secret_temp)))
            # print("MAE_cover:" + str(np.mean(mae_cover_temp)) + " MAE_secret:" + str(np.mean(mae_secret_temp)))

print("clamp total result:")
print("PSNR_Y_cover:" + str(np.mean(psnr_cover_y)) + " PSNR_Y_secret:" + str(np.mean(psnr_secret_y)))
print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))
print("MAE_cover:" + str(np.mean(mae_cover)) + " MAE_secret:" + str(np.mean(mae_secret)))
print("RMSE_cover:" + str(np.mean(rmse_cover)) + " RMSE_secret:" + str(np.mean(rmse_secret)))