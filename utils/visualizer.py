import os
import json
from typing import List, Dict, Optional

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 移除行尾的换行符
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data

def plot_run(run_dir, save_dir: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    metrics = load_jsonl(os.path.join(run_dir, "metrics.jsonl"))    # 加载指标数据
    batches = load_jsonl(os.path.join(run_dir, "batches.jsonl"))    # 加载批次数据
    if not metrics:
        return
    epochs = [m.get("epoch", 0) for m in metrics]
    losses = [m.get("loss", 0.0) for m in metrics]
    precisions = [m.get("precision", 0.0) for m in metrics]
    recalls = [m.get("recall", 0.0) for m in metrics]
    mious = [m.get("miou", 0.0) for m in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0][0]
    ax.plot(epochs, losses, marker="o", markersize=1)
    ax.set_title("Epoch Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax = axes[0][1]
    ax.plot(epochs, precisions, marker="o", markersize=1)
    ax.set_title("Precision@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax = axes[1][0]
    ax.plot(epochs, recalls, marker="o", markersize=1)
    ax.set_title("Recall@0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax = axes[1][1]
    ax.plot(epochs, mious, marker="o", markersize=1)
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

if __name__ == "__main__":
    import argparse
    import sys

    # 默认尝试在当前目录或父目录寻找 logs
    default_logs = "./logs"
    if not os.path.exists(default_logs) and os.path.exists("../logs"):
        default_logs = "../logs"

    parser = argparse.ArgumentParser(description="Generate training plots from log data.")
    parser.add_argument("--run_dir", type=str, help="Path to the specific run directory (e.g. logs/20231027_120000)")
    parser.add_argument("--log_dir", type=str, default=default_logs, help="Path to the logs root directory to search for latest run")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots (default: inside run_dir)")

    args = parser.parse_args()

    target_run = args.run_dir

    # 如果未指定具体 run_dir，则在 log_dir 中找最新的
    if not target_run:
        if not os.path.exists(args.log_dir):
            print(f"Error: Log directory '{args.log_dir}' not found. Please specify --run_dir or check your path.")
            sys.exit(1)

        subdirs = [os.path.join(args.log_dir, d) for d in os.listdir(args.log_dir) if os.path.isdir(os.path.join(args.log_dir, d))]
        # 简单过滤：只要目录下有 metrics.jsonl 就算有效
        valid_runs = [d for d in subdirs if os.path.exists(os.path.join(d, "metrics.jsonl"))]

        if not valid_runs:
            print(f"No valid run directories (containing metrics.jsonl) found in '{args.log_dir}'.")
            sys.exit(1)

        # 按文件夹名（通常是时间戳）或修改时间排序，取最新的
        # 这里假设文件夹名格式为 YYYYMMDD_HHMMSS，字典序排序即可；也可以用 getmtime
        target_run = max(valid_runs, key=os.path.getmtime)
        print(f"Auto-selected latest run: {target_run}")

    print(f"Plotting metrics for: {target_run}")
    plot_run(target_run, args.out)
    print(f"Plots saved to {args.out or target_run}")