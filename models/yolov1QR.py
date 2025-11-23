"""YOLOv1_QR 模型

简化版 YOLOv1 的特征网络与预测头，面向二维码检测任务。
- 默认网格大小 `S=4`，输出维度为 `S*S*5`（每个网格 5 个数：x, y, w, h, conf）。
- 模型设计为单框预测（每格一个框），以满足数据集标签定义与简易训练流程。

说明：
- 输入为形状 `[B, 3, H, W]` 的图像张量；
- 前向返回形状 `[B, S*S*5]` 的预测张量；
- 与 `utils.loss.yolo_v1_loss` 和 `predict.py` 保持一致的接口。
"""

import torch
import torch.nn as nn
from torch import Tensor

class ConvBlock(nn.Module):
    """卷积块：Conv2d + BatchNorm2d + LeakyReLU。

    通过 `padding=(kernel_size-1)//2` 保持空间尺寸。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),    # 0.1 负斜率，原地操作（不占用额外内存）
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class YOLOv1_QR(nn.Module):
    """简化 YOLOv1 的二维码检测模型。

    - 特征提取网络逐步下采样至 4x4 特征图；
    - 预测头展平特征后通过全连接得到 `4*4*5` 输出。
    """

    def __init__(self) -> None:
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
            nn.Linear(2048, 4 * 4 * 5),  # 输出层：4x4网格，每网格5个值（x,y,w,h,confidence）
            nn.Sigmoid()  # 将输出映射到(0,1)，确保confidence在(0,1)之间
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向计算。

        参数：
        - x: `[B, 3, H, W]`

        返回：
        - `[B, S*S*5]` 的预测向量（默认 S=4）
        """
        feat = self.features(x)
        out = self.pred(feat)
        return out

