import os
import logging
from datetime import datetime
import json

def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

class TrainingRecorder:
    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.batches_path = os.path.join(run_dir, "batches.jsonl")

    def record_epoch(self, epoch: int, loss: float, precision: float, recall: float, miou: float, lr: float | None = None) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": int(epoch),
                "loss": float(loss),
                "precision": float(precision),
                "recall": float(recall),
                "miou": float(miou),
                "lr": (None if lr is None else float(lr)),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, ensure_ascii=False) + "\n")

    def record_batch(self, epoch: int, batch_idx: int, loss: float) -> None:
        with open(self.batches_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": int(epoch),
                "batch": int(batch_idx),
                "loss": float(loss),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, ensure_ascii=False) + "\n")

def get_lr(optimizer) -> float:
    try:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    except Exception:
        return 0.0

def init_run_logger(base_log_dir: str, device, data_dir: str, img_size: int, S: int, batch_size: int, epochs: int):
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_log_dir, run_time)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    logger = setup_logger(run_dir)
    logger.info(f'[Config] device={device}, data_dir={data_dir}, img_size={img_size}, S={S}, batch_size={batch_size}, epochs={epochs}')
    recorder = TrainingRecorder(run_dir)
    return logger, run_dir, ckpt_dir, recorder
