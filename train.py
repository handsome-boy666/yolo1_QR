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

from models.yolov1QR import YOLOv1_QR
from utils.datasets import QRCodeDataset
from utils.loss import yolo_v1_loss


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
    ckpt_dir = str(tcfg.get('ckpt_dir'))
    return data_dir, img_size, S, batch_size, num_workers, epochs, learning_rate, ckpt_dir


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    """查找最新检查点"""
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not files:
        return None
    # 期望格式: epoch_XXX.pth
    files.sort()
    return os.path.join(ckpt_dir, files[-1])


def save_checkpoint(ckpt_dir: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """保存检查点"""
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    print(f'[Checkpoint] 已保存到: {path}')


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, S: int, epoch: int | None = None) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    iterator = tqdm(dataloader, desc=(f"Epoch {epoch}" if epoch is not None else "Training"), leave=False)
    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)  # [B, S*S*5]
        loss = yolo_v1_loss(preds, targets, S=S)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(dataloader))


def build_dataloader(data_dir, img_size, S, batch_size, num_workers, device: torch.device, if_train = True) -> DataLoader:
    """构建数据集与数据加载器"""
    dataset = QRCodeDataset(data_dir=data_dir, img_size=img_size, S=S, if_train=if_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )


def build_model_and_optimizer(device: torch.device, learning_rate: float) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """创建模型与优化器"""
    model = YOLOv1_QR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def maybe_resume(ckpt_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    """检查已有检查点并根据用户选择决定是否从最近检查点继续训练。

    返回开始的 epoch 序号。
    """
    ensure_dir(ckpt_dir)
    latest = find_latest_checkpoint(ckpt_dir)
    start_epoch = 1
    if latest is not None:
        print(f'[Resume] 检测到已有模型: {latest}')
        ans = input('检测到已有模型，是否继续训练(加载最近检查点)? (y/n): ').strip().lower()
        if ans == 'y':
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state['model_state'])
            try:
                optimizer.load_state_dict(state['optimizer_state'])
            except Exception:
                print('[Resume] 优化器状态加载失败，使用新优化器继续。')
            start_epoch = int(state.get('epoch', 0)) + 1
            print(f'[Resume] 从第 {start_epoch} 个 epoch 开始训练。')
        else:
            print('[Resume] 选择不继续训练，将从头开始。')
    return start_epoch


def run_training(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, S: int, start_epoch: int, epochs: int, ckpt_dir: str) -> None:
    """执行训练循环并在每个 epoch 保存检查点"""
    for epoch in range(start_epoch, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, S, epoch)
        print(f'Epoch [{epoch}/{epochs}] - loss: {avg_loss:.4f}')
        save_checkpoint(ckpt_dir, epoch, model, optimizer)


def main() -> None:
    # 加载配置
    data_dir, img_size, S, batch_size, num_workers, epochs, learning_rate, ckpt_dir = read_train_config('./config.yaml')

    device = get_device()
    print(f'[Config] device={device}, data_dir={data_dir}, img_size={img_size}, S={S}, batch_size={batch_size}, epochs={epochs}')

    # 数据集与加载器
    dataloader = build_dataloader(data_dir, img_size, S, batch_size, num_workers, device)

    # 模型与优化器
    model, optimizer = build_model_and_optimizer(device, learning_rate)

    # 断点续训选择
    start_epoch = maybe_resume(ckpt_dir, model, optimizer, device)

    # 训练循环
    run_training(model, dataloader, optimizer, device, S, start_epoch, epochs, ckpt_dir)


if __name__ == '__main__':
    main()