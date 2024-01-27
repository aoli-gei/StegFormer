import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import config
args = config.Args()

# 对数据集图像进行处理
transform = T.Compose([
    T.RandomCrop(128),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

# 使用 albumentations 库对图像进行处理
transform_A = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.augmentations.transforms.ChannelShuffle(0.3),
    ToTensorV2()
])

transform_A_valid = A.Compose([
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

transform_A_test = A.Compose([
    A.CenterCrop(width=1024, height=1024),
    ToTensorV2()
])

transform_A_test_256 = A.Compose([
    A.PadIfNeeded(min_width=256,min_height=256),
    A.CenterCrop(width=256, height=256),
    ToTensorV2()
])

DIV2K_path = "/home/whq135/dataset"
Flickr2K_path = "/home/whq135/dataset/Flickr2K"

batchsize = 12

# dataset


class DIV2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_train_HR"+"/*."+"png")))
        else:
            self.files = natsorted(
                sorted(glob.glob(DIV2K_path+"/DIV2K_valid_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item = trans_img['image']
        item = item/255.0
        return item

    def __len__(self):
        return len(self.files)

class Flickr2K_Dataset(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        self.mode = mode
        self.files = natsorted(
            sorted(glob.glob(Flickr2K_path+"/Flickr2K_HR"+"/*."+"png")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item= trans_img['image']
        item=item/255.0
        return item

    def __len__(self):
        return len(self.files)

class COCO_Test_Dataset(Dataset):
    def __init__(self, transforms_=None):
        self.transform = transforms_
        self.files = natsorted(
                sorted(glob.glob("/home/whq135/dataset/COCO2017/test2017"+"/*."+"jpg")))

    def __getitem__(self, index):
        img = cv2.imread(self.files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        trans_img = self.transform(image=img)
        item= trans_img['image']
        item=item/255.0
        return item

    def __len__(self):
        return len(self.files)

# dataloader
DIV2K_train_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_train_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)

DIV2K_val_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_val_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.single_batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

DIV2K_test_cover_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

DIV2K_test_secret_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

DIV2K_multi_train_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A, mode="train"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_val_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_valid, mode="val"),
    batch_size=args.multi_batch_iteration,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

DIV2K_multi_test_loader = DataLoader(
    DIV2K_Dataset(transforms_=transform_A_test, mode="val"),
    batch_size=args.test_multi_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_multi_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=args.test_multi_batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_cover_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

COCO_test_secret_loader = DataLoader(
    COCO_Test_Dataset(transforms_=transform_A_test_256),
    batch_size=1,
    shuffle=True,
    pin_memory=False,
    num_workers=1,
    drop_last=True
)

Flickr2K_multi_train_loader = DataLoader(
    Flickr2K_Dataset(transforms_=transform_A),
    batch_size=40,
    shuffle=True,
    pin_memory=True,
    num_workers=16,
    drop_last=True
)

Flickr2K_train_cover_loader = DataLoader(
    Flickr2K_Dataset(transforms_=transform_A),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)

Flickr2K_train_secret_loader = DataLoader(
    Flickr2K_Dataset(transforms_=transform_A),
    batch_size=batchsize,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)
