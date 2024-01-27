import torch
import torch.nn as nn
import torch.optim
import math
import numpy as np
from critic import *
from tensorboardX import SummaryWriter
from thop import profile
from datasets import *
from model import StegFormer
import os
import timm
import timm.scheduler
import config
from einops import rearrange
args = config.Args()

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# loss function
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class Restrict_Loss(nn.Module):
    """Restrict loss using L2 loss function"""

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, X):
        count1 = torch.sum(X > 1)
        count0 = torch.sum(X < 0)
        if count1 == 0:
            count1 = 1
        if count0 == 0:
            count0 = 1
        one = torch.ones_like(X)
        zero = torch.zeros_like(X)
        X_one = torch.where(X <= 1, 1, X)
        X_zero = torch.where(X >= 0, 0, X)
        diff_one = X_one-one
        diff_zero = zero-X_zero
        loss = torch.sum(0.5*(diff_one**2))/count1 + torch.sum(0.5*(diff_zero**2))/count0
        return loss

# 新建文件夹
model_version_name = args.model_name
save_path = args.path+'/checkpoint/'+model_version_name # 新建一个以模型版本名为名字的文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)

# tensorboard
writer = SummaryWriter(f'{args.path}/tensorboard_log/{args.model_name}/')

# StegFormer initiate
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
encoder.cuda()
decoder.cuda()

# loading model
model_path = ''
if args.train_next != 0:
    model_path = save_path + '/model_checkpoint_%.5i' % args.train_next + '.pt'
    state_dicts = torch.load(model_path)
    encoder.load_state_dict(state_dicts['encoder'], strict=False)
    decoder.load_state_dict(state_dicts['decoder'], strict=False)


# optimer and the learning rate scheduler
optim = torch.optim.AdamW([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=args.lr)
if args.train_next != 0:
    optim.load_state_dict(state_dicts['opt'])
scheduler = timm.scheduler.CosineLRScheduler(optimizer=optim,
                                             t_initial=args.epochs,
                                             lr_min=0,
                                             warmup_t=args.warm_up_epoch,
                                             warmup_lr_init=args.warm_up_lr_init)

# numbers of the parameter
with torch.no_grad():
    test_encoder_input = torch.randn(1, (args.num_secret+1)*3, args.image_size_train, args.image_size_train).to(args.device)
    test_decoder_input = torch.randn(1, 3, args.image_size_train, args.image_size_train).to(args.device)
    encoder_mac, encoder_params = profile(encoder, inputs=(test_encoder_input,))
    decoder_mac, decoder_params = profile(decoder, inputs=(test_decoder_input,))
    print("thop result:encoder FLOPs="+str(encoder_mac*2)+",encoder params="+str(encoder_params))
    print("thop result:decoder FLOPs="+str(decoder_mac*2)+",decoder params="+str(decoder_params))

# loss function
conceal_loss_function = L1_Charbonnier_loss().to(args.device)
reveal_loss_function = L1_Charbonnier_loss().to(args.device)
restrict_loss_funtion = Restrict_Loss().to(args.device)

# train
for i_epoch in range(args.epochs):
    sum_loss = []
    scheduler.step(i_epoch+args.train_next)
    for i_batch, img in enumerate(DIV2K_multi_train_loader):
        img = img.to(args.device)
        bs = args.multi_batch_szie
        cover = img[0:bs,:,:,:]
        secret = img[bs:,:,:,:]
        secret = rearrange(secret,'(b n) c h w -> b (n c) h w',n=args.num_secret)

        # encode
        msg = torch.cat([cover, secret], 1)
        encode_img = encoder(msg)  # 编码图像
        if args.norm_train == 'clamp':
            encode_img_c = torch.clamp(encode_img, 0, 1)
        else: 
            encode_img_c = encode_img

        # decode
        decode_img = decoder(encode_img_c)    # 解码图像

        # loss
        conceal_loss = conceal_loss_function(cover.cuda(), encode_img.cuda())
        reveal_loss = 2*reveal_loss_function(secret.cuda(), decode_img.cuda())
        total_loss = None
        if args.norm_train:
            restrict_loss = restrict_loss_funtion(encode_img.cuda())
            total_loss = conceal_loss + reveal_loss + restrict_loss
        else:
            total_loss = conceal_loss + reveal_loss
        sum_loss.append(total_loss.item())

        # backward
        total_loss.backward()
        optim.step()
        optim.zero_grad()
    # 进行验证，并记录指标
    if i_epoch % args.val_freq == 0:
        print("validation begin:")
        with torch.no_grad():
            # val
            encoder.eval()
            decoder.eval()

            # 评价指标
            psnr_secret = []
            psnr_cover = []
            ssim_secret = []
            ssim_cover = []

            # 在验证集上测试
            for img in DIV2K_multi_val_loader:
                img = img.to(args.device)
                bs = args.multi_batch_szie
                cover = img[0:bs,:,:,:]
                secret = img[bs:,:,:,:]
                secret_cat = rearrange(secret,'(b n) c h w -> b (n c) h w',n=args.num_secret)

                # encode
                msg = torch.cat([cover, secret_cat], 1)
                encode_img = encoder(msg)
                if args.norm_train == 'clamp':
                    encode_img = torch.clamp(encode_img, 0, 1)

                # decode
                decode_img = decoder(encode_img)
                decode_img = rearrange(decode_img,'b (n c) h w -> (b n) c h w',n=args.num_secret)

                # 计算各种指标
                # 拷贝进内存以方便计算
                cover = cover.cpu()
                secret = secret.cpu()
                encode_img = encode_img.cpu()
                decode_img = decode_img.cpu()

                psnr_encode_temp = calculate_psnr(cover, encode_img)
                psnr_decode_temp = calculate_psnr(secret, decode_img)
                psnr_cover.append(psnr_encode_temp)
                psnr_secret.append(psnr_decode_temp)

                ssim_encode_temp = calculate_ssim_skimage(cover, encode_img)
                ssim_decode_temp = calculate_ssim_skimage(secret, decode_img)
                ssim_cover.append(ssim_encode_temp)
                ssim_secret.append(ssim_decode_temp)
                writer.add_images('image/encode_img', encode_img, dataformats='NCHW', global_step=i_epoch+args.train_next)
                writer.add_images('image/decode_img', decode_img, dataformats='NCHW', global_step=i_epoch+args.train_next)

            # 写入 tensorboard
            writer.add_scalar("PSNR/PSNR_cover", np.mean(psnr_cover), i_epoch+args.train_next)
            writer.add_scalar("PSNR/PSNR_secret", np.mean(psnr_secret), i_epoch+args.train_next)
            writer.add_scalar("SSIM/SSIM_cover", np.mean(ssim_cover), i_epoch+args.train_next)
            writer.add_scalar("SSIM/SSIM_secret", np.mean(ssim_secret), i_epoch+args.train_next)
            print("PSNR_cover:" + str(np.mean(psnr_cover)) + " PSNR_secret:" + str(np.mean(psnr_secret)))

    # 绘制损失函数曲线
    print("epoch:"+str(i_epoch+args.train_next) + ":" + str(np.mean(sum_loss)))
    if i_epoch % 2 == 0:
        writer.add_scalar("loss", np.mean(sum_loss), i_epoch+args.train_next)

    # 保存当前模型以及优化器参数
    if (i_epoch % args.save_freq) == 0:
        torch.save({'opt': optim.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict()}, save_path + '/model_checkpoint_%.5i' % (i_epoch+args.train_next)+'.pt')


torch.save({'opt': optim.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()}, f'{save_path}/{model_version_name}.pt')
