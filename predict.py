"""YOLOv1_QR 预测脚本

功能：
- 单图预测并保存可视化与结果
- 测试集评估并输出 precision/recall/mIoU 及逐图预测

使用示例（Windows PowerShell）：
- 单图预测：
  python predict.py --config ./config.yaml --image ./dataset/images/test/sample.jpg --out ./predictions/single
- 测试集评估：
  python predict.py --config ./config.yaml --test

也可在 config.yaml 的 predict 段配置等价参数：
predict:
  image_path: ./dataset/images/test/sample.jpg
  out_dir: ./predictions/single
  run_test: true
"""
import os
import json
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from models.yolov1QR import YOLOv1_QR
from data.datasets import QRCodeDataset
from utils.utils import cell_to_bbox, DetectionMetrics
from models.utils import nms

def _sanitize_path(p: str | None) -> str:
    if not p:
        return ""
    s = str(p).strip()
    if (s.startswith('r"') and s.endswith('"')) or (s.startswith("r'") and s.endswith("'")):
        s = s[2:-1]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.replace("\\", "/")
    return os.path.normpath(s)

def read_predict_config(path: str) -> dict:
    """读取并解析 predict 段配置，返回统一字典；CLI 参数可覆盖此处配置。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    pcfg = cfg.get("predict", {})
    return {
        "ckpt_path": _sanitize_path(pcfg.get("ckpt_path") or ""),
        "data_dir": _sanitize_path(pcfg.get("data_dir", "./dataset")),
        "img_size": int(pcfg.get("img_size", 512)),
        "S": int(pcfg.get("S", 4)),
        "conf_thresh": float(pcfg.get("conf_thresh", 0.5)),
        "iou_thresh": float(pcfg.get("iou_thresh", 0.5)),
        "save_dir": _sanitize_path(pcfg.get("save_dir", "./predictions")),
        "image_path": _sanitize_path(pcfg.get("image_path") or ""),
        "out_dir": _sanitize_path(pcfg.get("out_dir", "./predictions/single")),
        "run_test": bool(pcfg.get("run_test", False)),
        "train_log_dir": _sanitize_path((cfg.get("train", {}) or {}).get("log_dir", "./logs")),
    }

def find_latest_checkpoint(search_dir: str) -> str | None:
    """在给定目录树中搜索最近修改的 .pth 权重，返回路径或 None。"""
    latest = None
    latest_mtime = -1.0
    if not os.path.isdir(search_dir):
        return None
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.endswith(".pth"):
                path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    mtime = 0.0
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest = path
    return latest

def load_model(ckpt_path: str | None, device: torch.device) -> YOLOv1_QR:
    """构建模型并按需加载权重；若未提供有效权重则返回随机初始化模型。"""
    model = YOLOv1_QR().to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state", state))
    return model

def build_transform(img_size: int):
    """与训练一致的图像预处理：Resize 到 `img_size` 并转为 Tensor。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def decode_boxes(pred: torch.Tensor, S: int, conf_thresh: float, iou_thresh: float):
    """将网络输出解码为归一化坐标框并执行 NMS，返回 [(box, score)] 列表。"""
    B = pred.size(0)
    pred = pred.view(B, S, S, 5).detach().cpu()
    boxes = []
    scores = []
    for j in range(S):
        for i in range(S):
            conf = float(pred[0, j, i, 4])
            if conf >= conf_thresh:
                x1, y1, x2, y2 = cell_to_bbox(i, j,
                                              float(pred[0, j, i, 0]),
                                              float(pred[0, j, i, 1]),
                                              float(pred[0, j, i, 2]),
                                              float(pred[0, j, i, 3]), S)
                boxes.append((x1, y1, x2, y2))
                scores.append(conf)
    keep = nms(boxes, scores, iou_thresh)
    return [(boxes[k], scores[k]) for k in keep]

def draw_boxes(image_path: str, boxes: list[tuple[tuple[float, float, float, float], float]], out_path: str):
    """在原图上绘制预测框并保存到 `out_path`。坐标为归一化，需要乘以图像尺寸转为像素。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font_size = max(12, int(min(w, h) * 0.02))
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), score in boxes:
        x1p = int(x1 * w)
        y1p = int(y1 * h)
        x2p = int(x2 * w)
        y2p = int(y2 * h)
        draw.rectangle([x1p, y1p, x2p, y2p], outline=(255, 0, 0), width=2)
        label = f"{score:.2f}"
        ty = y1p - 12 if (y1p - 12) > 0 else (y1p + 2)
        tx = x1p + 2
        try:
            bbox = draw.textbbox((tx, ty), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(label, font=font)
        draw.rectangle([tx - 1, ty - 1, tx + tw + 1, ty + th + 1], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 0), font=font)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)

def save_txt(out_path: str, boxes: list[tuple[tuple[float, float, float, float], float]]):
    """保存预测结果到文本：每行 `conf x1 y1 x2 y2`（均为归一化坐标）。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for (x1, y1, x2, y2), score in boxes:
            f.write(f"{score:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")

def run_single(cfg: dict, image_path: str | None, out_dir: str | None, device: torch.device):
    """单图预测：加载图片、推理、解码与 NMS，并保存可视化与结果文件。"""
    image_path = image_path or cfg["image_path"]
    out_dir = out_dir or cfg["out_dir"]
    if not image_path:
        raise SystemExit("image_path 未提供")
    if not os.path.isfile(image_path):
        raise SystemExit(f"图片不存在: {image_path}")
    ckpt_path = cfg["ckpt_path"] or find_latest_checkpoint(cfg["train_log_dir"]) or ""
    if not ckpt_path:
        raise SystemExit("未找到权重文件，请在 predict.ckpt_path 指定或训练后再试")
    model = load_model(ckpt_path, device)
    model.eval()
    tfm = build_transform(cfg["img_size"])
    img = Image.open(image_path).convert("RGB")
    img_t = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_t)
    boxes = decode_boxes(pred, cfg["S"], cfg["conf_thresh"], cfg["iou_thresh"])
    base = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(out_dir, f"{base}_pred.jpg")
    txt_path = os.path.join(out_dir, f"{base}.txt")
    draw_boxes(image_path, boxes, vis_path)
    save_txt(txt_path, boxes)

def run_test(cfg: dict, device: torch.device):
    """测试集评估：遍历 `images/test`，累积 DetectionMetrics 并保存逐图预测与总体指标。"""
    ckpt_path = cfg["ckpt_path"] or find_latest_checkpoint(cfg["train_log_dir"]) or ""
    if not ckpt_path:
        raise SystemExit("未找到权重文件，请在 predict.ckpt_path 指定或训练后再试")
    model = load_model(ckpt_path, device)
    model.eval()
    ds = QRCodeDataset(data_dir=cfg["data_dir"], img_size=cfg["img_size"], S=cfg["S"], if_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    metrics = DetectionMetrics(iou_thresh=cfg["iou_thresh"], conf_thresh=cfg["conf_thresh"])
    save_base = os.path.join(cfg["save_dir"], "test")
    preds_dir = os.path.join(save_base, "preds")
    vis_dir = os.path.join(save_base, "vis")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for (imgs, targets) in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            pred = model(imgs)
            metrics.update(pred, targets, cfg["S"]) 
            boxes = decode_boxes(pred, cfg["S"], cfg["conf_thresh"], cfg["iou_thresh"]) 
            img_path, _ = ds.samples[count]
            base = os.path.splitext(os.path.basename(img_path))[0]
            save_txt(os.path.join(preds_dir, f"{base}.txt"), boxes)
            draw_boxes(img_path, boxes, os.path.join(vis_dir, f"{base}_pred.jpg"))
            count += 1
    precision, recall, miou = metrics.compute()
    metrics_path = os.path.join(save_base, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"precision": precision, "recall": recall, "mIoU": miou, "count": count}, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv1_QR 推理与测试评估",
        epilog=(
            "示例:\n"
            "  单图预测: python predict.py --config ./config.yaml --image ./dataset/images/test/sample.jpg --out ./predictions/single\n"
            "  测试评估: python predict.py --config ./config.yaml --test"
        ),
    )
    parser.add_argument("--config", type=str, default="./config.yaml", help="配置文件路径")
    parser.add_argument("--image", type=str, default=None, help="单图预测的图片路径，若省略则使用配置中的 image_path")
    parser.add_argument("--out", type=str, default=None, help="单图预测输出目录，若省略则使用配置中的 out_dir")
    parser.add_argument("--test", action="store_true", help="运行测试集评估（忽略 --image/--out）")
    args = parser.parse_args()
    cfg = read_predict_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.test or cfg["run_test"]:
        run_test(cfg, device)
        return
    image_path = args.image if args.image is not None else (cfg.get("image_path") or None)
    out_dir = args.out if args.out is not None else (cfg.get("out_dir") or None)
    run_single(cfg, image_path, out_dir, device)

if __name__ == "__main__":
    main()
