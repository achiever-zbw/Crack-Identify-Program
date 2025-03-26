import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split

from model.unet_parts import *
from model.unet_model import UNet
from dataset import *

train_dir_img = r"D:\裂缝数据集\CrackDataset-main\用来训练的图片集合"
train_dir_mask = r"D:\裂缝数据集\CrackDataset-main\用来训练的mask集合"
val_dir_img = r"D:\裂缝数据集\CrackDataset-main\CFD\val"
val_dir_mask = r"D:\裂缝数据集\CrackDataset-main\CFD\val_label"


def train_model(
    model,  # 模型
    device,  # 设备
    epoches=5,
    batch_size=5,  # 批次，怕爆显存所以小点吧...
    learning_rate=1e-5,  # 学习率
    weight_decay=1e-8,
):
    # 创建数据集
    train_dataset = DatasetFunc(train_dir_img, train_dir_mask)  # 训练
    val_dataset = DatasetFunc(val_dir_img, val_dir_mask)  # 验证

    # 创建数据加载器
    dataloader = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader)
    val_loader = DataLoader(val_dataset, shuffle=True, **dataloader)

    # 初始化加载器
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 交叉熵损失
    criterion = nn.BCEWithLogitsLoss()

    # 训练
    for epoch in range(1, epoches+1):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            images, masks = batch
            # 转移到设备
            images = images.to(device)
            masks = masks.unsqueeze(1).float().to(device)  # 增加通道维度 [B,1,H,W]

            # 向前传播
            output = model(images)
            loss = criterion(output, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 优化器
            optimizer.step()
            # 加和损失
            epoch_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                # 转移到设备
                images = images.to(device)
                masks = masks.unsqueeze(1).float().to(
                    device)  # 增加通道维度 [B,1,H,W]
                output = model(images)
                loss = criterion(output, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch}/{epoches}')
        print(f'平均训练损失: {avg_train_loss:.4f}')
        print(f'平均验证损失: {avg_val_loss:.4f}')


def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    torch.manual_seed(42)

    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化模型
    model = UNet(n_channels=1, n_classes=1)  # 单通道输入，单通道输出
    model = model.to(device)

    # 训练
    train_model(
        model=model,
        device=device,
        epoches=50,
        batch_size=4,
        learning_rate=1e-4,
        weight_decay=1e-8
    )

    # 保存最终模型
    torch.save(model.state_dict(), 'checkpoints/final_model_1.pth')


if __name__ == '__main__':
    main()
