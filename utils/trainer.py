import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.loss import yolo_v1_loss
from utils.utils import DetectionMetrics
from utils.logger import get_lr


def maybe_resume(search_dir, model, optimizer, device, logger) -> int:
    os.makedirs(search_dir, exist_ok=True)
    if not os.path.isdir(search_dir):
        latest = None
    else:
        latest = None
        latest_mtime = -1.0
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith('.pth'):
                    path = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(path)
                    except Exception:
                        mtime = 0.0
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = path

    start_epoch = 1
    if latest is not None:
        logger.info(f'[Resume] 检测到已有模型: {latest}')
        ans = input('检测到已有模型，是否继续训练(加载最近检查点)? (y/n): ').strip().lower()
        if ans == 'y':
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state['model_state'])
            try:
                optimizer.load_state_dict(state['optimizer_state'])
            except Exception:
                logger.info('[Resume] 优化器状态加载失败，使用新优化器继续。')
            start_epoch = int(state.get('epoch', 0)) + 1
            logger.info(f'[Resume] 从第 {start_epoch} 个 epoch 开始训练。')
        else:
            logger.info('[Resume] 选择不继续训练，将从头开始。')
    return start_epoch


def save_checkpoint(ckpt_dir, epoch: int, model, optimizer, logger) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
    logger.info(f'[Checkpoint] 已保存到: {path}')


def train_one_epoch(model, dataloader, optimizer, device, S, epoch: Optional[int] = None, metrics: Optional[DetectionMetrics] = None, recorder=None) -> float:
    model.train()
    total_loss = 0.0
    iterator = tqdm(dataloader, desc=(f"Epoch {epoch}" if epoch is not None else "Training"), leave=False)
    for batch_idx, (images, targets) in enumerate(iterator):
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        loss = yolo_v1_loss(preds, targets, S=S)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if metrics is not None:
            metrics.update(preds.detach(), targets.detach(), S)
        if recorder is not None:
            recorder.record_batch(int(epoch) if epoch is not None else 0, int(batch_idx), float(loss.item()))
        total_loss += loss.item()
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, len(dataloader))


def run_training(model, dataloader, optimizer, device, S, start_epoch, epochs, ckpt_dir, logger, recorder) -> None:
    for epoch in range(start_epoch, epochs + 1):
        metrics = DetectionMetrics(iou_thresh=0.5, conf_thresh=0.5)
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, S, epoch, metrics, recorder)
        precision, recall, miou = metrics.compute()
        logger.info(f'Epoch [{epoch}/{epochs}] - loss: {avg_loss:.4f}, precision@0.5: {precision:.3f}, recall@0.5: {recall:.3f}, mIoU: {miou:.3f}')
        if recorder is not None:
            recorder.record_epoch(epoch, avg_loss, precision, recall, miou, get_lr(optimizer))
        save_checkpoint(ckpt_dir, epoch, model, optimizer, logger)