"""二维码数据集封装

自动匹配 `images` 与 `labels`（支持 `train` 子目录），仅收集合法图片，按同名配对标签。
标签格式遵循常见 YOLO 格式：`class x y w h`（均为 [0,1] 范围）。
输出目标张量形状为 `(S, S, 5)`：`(x_offset, y_offset, w, h, conf)`。
"""

import os
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class QRCodeDataset(Dataset):
    """二维码数据集。

    参数：
    - data_dir: 数据根目录，包含 `images/` 与 `labels/`（可含 `train/` 子目录）。
    - img_size: 训练/推理的输入尺寸（会进行 `Resize`）。
    - S: 网格大小（默认 4）。
    - transform: 自定义 `torchvision` 变换，若为 `None` 则使用默认（`Resize+ToTensor`）。
    """

    def __init__(self, data_dir, img_size = 512, S = 4, if_train = True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.S = S

        image_root = os.path.join(data_dir, 'images')
        label_root = os.path.join(data_dir, 'labels')

        if os.path.isdir(image_root) and os.path.isdir(label_root):
            if if_train:
                self.image_dir = os.path.join(image_root, 'train')
                self.label_dir = os.path.join(label_root, 'train')
            else:
                self.image_dir  = os.path.join(image_root, 'test')
                self.label_dir = os.path.join(label_root, 'test')
        else:
            raise FileNotFoundError("数据集目录不存在")

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in os.listdir(self.image_dir) if os.path.splitext(f)[1].lower() in img_exts]    # 收集所有图片文件
        label_names = {os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.lower().endswith('.txt')}    # 收集所有标签文件（不包含扩展名）

        self.samples: List[Tuple[str, str]] = []
        for img_file in sorted(image_files):
            name = os.path.splitext(img_file)[0]
            if name in label_names:
                self.samples.append((
                    os.path.join(self.image_dir, img_file),
                    os.path.join(self.label_dir, name + '.txt')
                ))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"在 '{self.image_dir}' 与 '{self.label_dir}' 下未找到配对的图片与标签文件")

        # 构建数据集变换
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img_path, label_path = self.samples[idx]

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"图片不存在或不是文件: {img_path}")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        target = torch.zeros(self.S, self.S, 5) #(S, S, 5) 网格，每个单元格包含 (x_offset, y_offset, w, h, conf)

        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()    # 分割行内容为列表
                    if len(parts) != 5:
                        continue
                    try:
                        _, x, y, w, h = map(float, parts)    # 解析类别、中心坐标、宽高
                    except Exception:
                        continue
                    # 过滤非有限值，避免 NaN/Inf
                    if not (torch.isfinite(torch.tensor([x, y, w, h])).all()):
                        continue
                    # 夹紧到有效范围（避免 x==1 导致 i==S 越界）
                    eps = 1e-6
                    x = float(max(0.0, min(1.0 - eps, x)))  # 夹紧到 [0, 1-eps]
                    y = float(max(0.0, min(1.0 - eps, y)))  # 夹紧到 [0, 1-eps]
                    w = float(max(eps, min(1.0, w)))  # 夹紧到 [eps, 1]
                    h = float(max(eps, min(1.0, h)))  # 夹紧到 [eps, 1]
                    i = int(self.S * x) 
                    j = int(self.S * y)
                    if 0 <= i < self.S and 0 <= j < self.S:
                        target[j, i, 0] = self.S * x - i
                        target[j, i, 1] = self.S * y - j
                        target[j, i, 2] = w
                        target[j, i, 3] = h
                        target[j, i, 4] = 1.0
        return img, target
