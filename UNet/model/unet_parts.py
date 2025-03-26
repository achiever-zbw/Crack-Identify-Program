""" UNet 模型组件
1、双卷积
2、下采样
3、上采样
4、输出"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积模块，进行两次卷积操作
    1、padding 保证输入和最终输出的特征图大小一致
    2、加入 BN 与 ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):  # 初始化
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块，包含一次的最大池化（尺寸减半 + 2 × 卷积）
    运用双卷积模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(  # 最大池化 + 双卷积
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块，使用双线性插值和转置卷积扩大特征图"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        in_channels(int) : 输入通道
        out_channels(iny) : 输出通道
        bilinear(bool) : 方式选择
        """
        super().__init__()

        if bilinear:  # 双线性插值
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:  # 转置卷积
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样
        # 调整尺寸并拼接
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 对称填充，保证尺寸对齐
        x = torch.cat([x2, x1], dim=1)  # 编码器的细粒度特征 (x2) 与解码器的语义特征 (x1) 融合
        return self.conv(x)


class OutConv(nn.Module):
    """输出层，包含一层卷积，输出为目标类别 -- 降维"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
