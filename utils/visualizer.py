import os
import json
from typing import List, Dict, Optional

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data

def plot_run(run_dir: str, save_dir: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    metrics = load_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    batches = load_jsonl(os.path.join(run_dir, "batches.jsonl"))
    if not metrics:
        return
    epochs = [m.get("epoch", 0) for m in metrics]
    losses = [m.get("loss", 0.0) for m in metrics]
    precisions = [m.get("precision", 0.0) for m in metrics]
    recalls = [m.get("recall", 0.0) for m in metrics]
    mious = [m.get("miou", 0.0) for m in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0][0]
    ax.plot(epochs, losses, marker="o")
    ax.set_title("Epoch Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax = axes[0][1]
    ax.plot(epochs, precisions, marker="o")
    ax.set_title("Precision@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax = axes[1][0]
    ax.plot(epochs, recalls, marker="o")
    ax.set_title("Recall@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax = axes[1][1]
    ax.plot(epochs, mious, marker="o")
    ax.set_title("mIoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    plt.tight_layout()
    out_dir = save_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "metrics.png"))
    plt.close(fig)
    if batches:
        by_epoch: Dict[int, List[Dict]] = {}
        for b in batches:
            e = int(b.get("epoch", 0))
            by_epoch.setdefault(e, []).append(b)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        xs = []
        ys = []
        for e in sorted(by_epoch.keys()):
            for b in sorted(by_epoch[e], key=lambda x: x.get("batch", 0)):
                xs.append((e, int(b.get("batch", 0))))
                ys.append(float(b.get("loss", 0.0)))
        x_ticks = list(range(len(xs)))
        ax2.plot(x_ticks, ys)
        ax2.set_title("Batch Loss")
        ax2.set_xlabel("Batch Index (by epoch)")
        ax2.set_ylabel("Loss")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "batches.png"))
        plt.close(fig2)