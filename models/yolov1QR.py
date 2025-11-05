"""
修改了YOLOv1的模型结构，用于检测二维码
* S：网格划分数=4
* B：每个网格预测框数量=2
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):  # 卷积块（卷积层+批量归一化+LeakyReLU）

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()  # 初始化父类
        if padding is None:  # 确保padding不为None，输出的尺寸保持不变
            padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),  # 卷积层
            nn.BatchNorm2d(out_channels),  # 批量归一化
            nn.LeakyReLU(0.1, inplace=True)  # LeakyReLU激活函数，负斜率为0.1
        )

    def forward(self, x):
        return self.block(x)  # 前向传播：通过卷积块


class YOLOv1_QR(nn.Module):  # YOLOv1模型（二维码检测）
    def __init__(self):
        super().__init__()
        # 特征提取网络：逐步下采样并提取图像特征
        self.features = nn.Sequential(
            ConvBlock(3, 64, 7, stride=2, padding=3),  # 输入3通道，输出64通道，7x7卷积，步长2，填充3
            nn.MaxPool2d(2, 2),  # 2x2最大池化，下采样

            ConvBlock(64, 192, 3),  # 输入64通道，输出192通道，3x3卷积
            nn.MaxPool2d(2, 2),  # 2x2最大池化，下采样

            ConvBlock(192, 128, 1),  # 1x1卷积，降维
            ConvBlock(128, 256, 3),  # 3x3卷积，升维
            ConvBlock(256, 256, 1),  # 1x1卷积，降维
            ConvBlock(256, 512, 3),  # 3x3卷积，升维
            nn.MaxPool2d(2, 2),  # 2x2最大池化，下采样

            # 重复4次：1x1卷积降维 + 3x3卷积升维，保持512通道
            *[ConvBlock(512, 256, 1), ConvBlock(256, 512, 3)] * 4,
            ConvBlock(512, 512, 1),  # 1x1卷积，降维
            ConvBlock(512, 1024, 3),  # 3x3卷积，升维至1024
            nn.MaxPool2d(2, 2),  # 2x2最大池化，下采样

            # 重复2次：1x1卷积降维 + 3x3卷积升维，保持1024通道
            *[ConvBlock(1024, 512, 1), ConvBlock(512, 1024, 3)] * 2,
            ConvBlock(1024, 1024, 3),  # 3x3卷积，保持1024通道
            ConvBlock(1024, 1024, 3, stride=2),  # 3x3卷积，步长2，进一步下采样

            ConvBlock(1024, 1024, 3),  # 3x3卷积，保持1024通道
            ConvBlock(1024, 1024, 3),  # 3x3卷积，保持1024通道
            nn.MaxPool2d(2, 2),  # 2x2最大池化，最终下采样至4x4特征图
        )

        # 预测头：将特征图展平后通过全连接层输出预测结果
        self.pred = nn.Sequential(
            nn.Flatten(),  # 展平特征图：4x4x1024 = 16384
            nn.Linear(4 * 4 * 1024, 2048),  # 全连接层，降维至2048
            nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
            nn.Dropout(0.5),  # Dropout，防止过拟合
            nn.Linear(2048, 4 * 4 * 5)  # 输出层：4x4网格，每网格5个值（x,y,w,h,confidence）
        )

    def forward(self, x):
        feat = self.features(x)  # 特征提取
        out = self.pred(feat)  # 预测输出
        return out  # 返回预测结果

