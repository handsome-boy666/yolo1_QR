import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class QRCodeDataset(Dataset):
    """
    自定义数据集：自动识别 train 子目录或根目录，仅收集图片文件并按同名标签配对。
    输出标签张量形状为 (S, S, 5): [x_offset, y_offset, w, h, conf]
    """
    def __init__(self, data_dir, img_size=512, S=4, transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.S = S

        image_root = os.path.join(data_dir, 'images')
        label_root = os.path.join(data_dir, 'labels')
        image_dir_train = os.path.join(image_root, 'train')
        label_dir_train = os.path.join(label_root, 'train')

        if os.path.isdir(image_dir_train) and os.path.isdir(label_dir_train):
            self.image_dir = image_dir_train
            self.label_dir = label_dir_train
        else:
            self.image_dir = image_root
            self.label_dir = label_root

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in os.listdir(self.image_dir) if os.path.splitext(f)[1].lower() in img_exts]
        label_names = {os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.lower().endswith('.txt')}

        self.samples = []
        for img_file in sorted(image_files):
            name = os.path.splitext(img_file)[0]
            if name in label_names:
                self.samples.append((
                    os.path.join(self.image_dir, img_file),
                    os.path.join(self.label_dir, name + '.txt')
                ))

        if len(self.samples) == 0:
            raise FileNotFoundError(f"在 '{self.image_dir}' 与 '{self.label_dir}' 下未找到配对的图片与标签文件")

        self.transform = transform or transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"图片不存在或不是文件: {img_path}")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        target = torch.zeros(self.S, self.S, 5)

        if os.path.isfile(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        _, x, y, w, h = map(float, parts)
                    except Exception:
                        continue
                    # 过滤非有限值，避免 NaN/Inf
                    if not (torch.isfinite(torch.tensor([x, y, w, h])).all()):
                        continue
                    # 夹紧到有效范围（避免 x==1 导致 i==S 越界）
                    eps = 1e-6
                    x = float(max(0.0, min(1.0 - eps, x)))
                    y = float(max(0.0, min(1.0 - eps, y)))
                    w = float(max(eps, min(1.0, w)))
                    h = float(max(eps, min(1.0, h)))
                    i = int(self.S * x)
                    j = int(self.S * y)
                    if 0 <= i < self.S and 0 <= j < self.S:
                        target[j, i, 0] = self.S * x - i
                        target[j, i, 1] = self.S * y - j
                        target[j, i, 2] = w
                        target[j, i, 3] = h
                        target[j, i, 4] = 1.0
        return img, target
