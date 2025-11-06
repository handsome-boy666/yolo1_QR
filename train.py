import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolov1QR import YOLOv1_QR
from utils.datasets import QRCodeDataset
from utils.loss import yolo_v1_loss


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_latest_checkpoint(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not files:
        return None
    # 期望格式: epoch_XXX.pth
    files.sort()
    return os.path.join(ckpt_dir, files[-1])


def save_checkpoint(ckpt_dir: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    print(f'[Checkpoint] 已保存到: {path}')


def train_one_epoch(model, dataloader, optimizer, device, S, epoch=None):
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


def main():
    # 加载配置
    cfg = load_config('./config.yaml')
    tcfg = cfg.get('train', {})
    data_dir = tcfg.get('data_dir', './data')
    img_size = int(tcfg.get('img_size', 512))
    S = int(tcfg.get('S', 4))
    batch_size = int(tcfg.get('batch_size', 16))
    num_workers = int(tcfg.get('num_workers', 0))
    epochs = int(tcfg.get('epochs', 50))

    device = get_device()
    print(f'[Config] device={device}, data_dir={data_dir}, img_size={img_size}, S={S}, batch_size={batch_size}, epochs={epochs}')

    # 数据集与加载器
    dataset = QRCodeDataset(data_dir=data_dir, img_size=img_size, S=S)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # 模型与优化器
    model = YOLOv1_QR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 检查已有模型，询问是否继续训练
    ckpt_dir = os.path.join('.', 'checkpoints')
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

    # 训练循环
    for epoch in range(start_epoch, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, S, epoch)
        print(f'Epoch [{epoch}/{epochs}] - loss: {avg_loss:.4f}')

        # 每个 epoch 保存模型
        save_checkpoint(ckpt_dir, epoch, model, optimizer)


if __name__ == '__main__':
    main()