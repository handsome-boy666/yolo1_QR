"""训练脚本

封装训练流程：读取配置、数据加载、损失计算、优化器更新与检查点保存。
满足以下要求：
- 每个 epoch 保存模型；
- 运行前检查是否有检查点并询问是否继续训练；
- tqdm 进度条显示 batch 级损失；
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import init_run_logger
from utils.trainer import maybe_resume, run_training

from models.yolov1QR import YOLOv1_QR
from data.datasets import QRCodeDataset
from models.loss import yolo_v1_loss


def read_train_config(path: str) -> tuple[str, int, int, int, int, int, float, str]:
    """读取并解析训练配置。

    返回值依次为:
    data_dir, img_size, S, batch_size, num_workers, epochs, learning_rate, ckpt_dir
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg.get('train', {})
    data_dir = tcfg.get('data_dir')
    img_size = int(tcfg.get('img_size'))
    S = int(tcfg.get('S'))
    batch_size = int(tcfg.get('batch_size'))
    num_workers = int(tcfg.get('num_workers'))
    epochs = int(tcfg.get('epochs'))
    learning_rate = float(tcfg.get('learning_rate'))
    log_dir = str(tcfg.get('log_dir'))
    return data_dir, img_size, S, batch_size, num_workers, epochs, learning_rate, log_dir



def main() -> None:
    # 加载配置
    data_dir, img_size, S, batch_size, num_workers, epochs, learning_rate, log_dir = read_train_config('./config.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger, run_dir, ckpt_dir, recorder = init_run_logger(log_dir, device, data_dir, img_size, S, batch_size, epochs)

    # 数据集与加载器
    dataset = QRCodeDataset(data_dir=data_dir, img_size=img_size, S=S, if_train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # 模型与优化器
    model = YOLOv1_QR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 断点续训选择
    start_epoch = maybe_resume(log_dir, model, optimizer, device, logger)

    # 训练循环
    run_training(model, dataloader, optimizer, device, S, start_epoch, epochs, ckpt_dir, logger, recorder)


if __name__ == '__main__':
    main()
