import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torchvision import transforms


def threshold_transform(x):
    """将图像转换为二值掩码"""
    return (x > 0.5).long().squeeze(0)


class RandomTransforms:
    """同步空间变换（图像双线性插值，掩码最近邻插值）"""

    def __init__(self, size):
        self.img_transform = transforms.Compose([
            transforms.RandomRotation(30, interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=Image.BILINEAR)
        ])
        self.mask_transform = transforms.Compose([
            transforms.RandomRotation(30, interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=Image.NEAREST)
        ])

    def __call__(self, img, mask):
        """设置随机种子使 mask 与 image 进行一致的变换"""
        seed = np.random.randint(0, 2**31)
        torch.manual_seed(seed)
        img = self.img_transform(img)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)
        return img, mask


class DatasetFunc(Dataset):
    """安全配对文件名的数据集类"""

    def __init__(self, img_dir, mask_dir, size=256):
        # 正确配对文件名
        self.pairs = []
        for img_file in os.listdir(img_dir):
            base = os.path.splitext(img_file)[0]
            mask_file = f"{base}.png"
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                self.pairs.append((
                    os.path.join(img_dir, img_file),
                    mask_path
                ))

        self.random_transforms = RandomTransforms(size)
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485], std=[0.229])
        ])
        self.mask_transform = Compose([
            ToTensor(),
            Lambda(threshold_transform)
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        img, mask = self.random_transforms(img, mask)
        return self.img_transform(img), self.mask_transform(mask)