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
from models.utils import nms, box_iou
import matplotlib
matplotlib.use("Agg")  # 设置非交互式后端，避免 Qt 报错
import matplotlib.pyplot as plt

def read_predict_config(path: str) -> dict:
    """读取并解析 predict 段配置，返回统一字典；CLI 参数可覆盖此处配置。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    pcfg = cfg.get("predict", {})
    return {
        "ckpt_path": pcfg.get("ckpt_path") or "",
        "data_dir": pcfg.get("data_dir", "./dataset"),
        "img_size": int(pcfg.get("img_size", 512)),
        "S": int(pcfg.get("S", 4)),
        "conf_thresh": float(pcfg.get("conf_thresh", 0.5)),
        "iou_thresh": float(pcfg.get("iou_thresh", 0.5)),
        "save_dir": pcfg.get("save_dir", "./predictions"),
        "image_path": pcfg.get("image_path") or "",
        "out_dir": pcfg.get("out_dir", "./predictions/single"),
        "run_test": bool(pcfg.get("run_test", False)),
        "train_log_dir": pcfg.get("train_log_dir", "./logs"),
        "test_batch_size": int(pcfg.get("test_batch_size", 64)),
    }

def load_model(ckpt_path: str | None, device: torch.device) -> YOLOv1_QR:
    """构建模型并按需加载权重；若未提供有效权重则返回随机初始化模型。"""
    model = YOLOv1_QR().to(device)
    if ckpt_path and os.path.isfile(ckpt_path): # 检查权重文件是否存在
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state", state))
    return model

def build_transform(img_size: int):
    """与训练一致的图像预处理：Resize 到 `img_size` 并转为 Tensor。"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def decode_boxes(pred: torch.Tensor, S: int, conf_thresh: float, iou_thresh: float, sample_idx: int = 0):
    """将网络输出解码为归一化坐标框，返回 NMS 后和 NMS 前的 [(box, score)] 列表。
    
    Returns:
        nms_boxes: list of ((x1, y1, x2, y2), score) after NMS
        raw_boxes: list of ((x1, y1, x2, y2), score) before NMS
    """
    B = pred.size(0)
    pred = pred.view(B, S, S, 5).detach().cpu()
    boxes = []
    scores = []
    for j in range(S):
        for i in range(S):
            conf = float(pred[int(sample_idx), j, i, 4])
            if conf >= conf_thresh:
                x1, y1, x2, y2 = cell_to_bbox(i, j,
                                              float(pred[int(sample_idx), j, i, 0]),
                                              float(pred[int(sample_idx), j, i, 1]),
                                              float(pred[int(sample_idx), j, i, 2]),
                                              float(pred[int(sample_idx), j, i, 3]), S)
                boxes.append((x1, y1, x2, y2))
                scores.append(conf)
    
    raw_boxes = [(boxes[i], scores[i]) for i in range(len(boxes))]
    keep = nms(boxes, scores, iou_thresh)
    nms_boxes = [(boxes[k], scores[k]) for k in keep]
    return nms_boxes, raw_boxes

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
        draw.rectangle([x1p, y1p, x2p, y2p], outline=(255, 0, 0), width=3)
        label = f"QR:{score:.2f}"
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

def compute_ap(det_boxes: list, gt_boxes: dict, iou_threshold: float = 0.5) -> tuple[float, list, list]:
    """计算 AP 以及 Precision-Recall 曲线数据。
    
    Args:
        det_boxes: list of (img_idx, conf, x1, y1, x2, y2)
        gt_boxes: dict {img_idx: [(x1, y1, x2, y2), ...]}
        iou_threshold: IoU 阈值
    
    Returns:
        ap: Average Precision
        precisions: list of precision values
        recalls: list of recall values
    """
    # 按置信度降序排列
    det_boxes.sort(key=lambda x: x[1], reverse=True)
    
    tp = torch.zeros(len(det_boxes))
    fp = torch.zeros(len(det_boxes))
    
    total_gt = sum(len(gts) for gts in gt_boxes.values())
    if total_gt == 0:
        return 0.0, [], []
        
    # 记录每个 GT 是否被匹配过
    gt_matched = {k: torch.zeros(len(v)) for k, v in gt_boxes.items()}
    
    for idx, (img_idx, conf, x1, y1, x2, y2) in enumerate(det_boxes):
        gts = gt_boxes.get(img_idx, [])
        pred_box = (x1, y1, x2, y2)
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(gts):
            iou = box_iou(pred_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou > iou_threshold:
            if gt_matched[img_idx][best_gt_idx] == 0:
                tp[idx] = 1
                gt_matched[img_idx][best_gt_idx] = 1
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1
            
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / (total_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # 添加起始点 (Recall=0, Precision=1)
    precisions = torch.cat((torch.tensor([1.0]), precisions))
    recalls = torch.cat((torch.tensor([0.0]), recalls))
    
    # 计算 AP (Area Under Curve)
    # 使用梯形法则，或者简单的 P*dR
    # 这里使用简单的 trapz, 计算积分面积
    ap = torch.trapz(precisions, recalls).item()
    
    return ap, precisions.tolist(), recalls.tolist()

def plot_pr_curve(precisions: list, recalls: list, ap: float, save_path: str):  # 绘制 Precision-Recall 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.4f})")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.savefig(save_path)
    plt.close()

def run_single(cfg: dict, image_path: str | None, out_dir: str | None, device: torch.device):
    """单图预测：加载图片、推理、解码与 NMS，并保存可视化与结果文件。"""
    image_path = image_path or cfg["image_path"]
    out_dir = out_dir or cfg["out_dir"]
    if not image_path:
        raise SystemExit("image_path 未提供")
    if not os.path.isfile(image_path):
        raise SystemExit(f"图片不存在: {image_path}")
    ckpt_path = cfg["ckpt_path"] or ""
    if not ckpt_path:
        raise SystemExit("未找到权重文件，请在 predict.ckpt_path 指定或训练后再试")
    model = load_model(ckpt_path, device)
    model.eval()
    tfm = build_transform(cfg["img_size"])
    img = Image.open(image_path).convert("RGB")
    img_t = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_t)
    boxes, raw_boxes = decode_boxes(pred, cfg["S"], cfg["conf_thresh"], cfg["iou_thresh"])
    base = os.path.splitext(os.path.basename(image_path))[0]
    vis_path = os.path.join(out_dir, f"{base}_pred.jpg")
    txt_path = os.path.join(out_dir, f"{base}.txt")
    raw_txt_path = os.path.join(out_dir, f"{base}_raw.txt")
    draw_boxes(image_path, boxes, vis_path)
    save_txt(txt_path, boxes)
    save_txt(raw_txt_path, raw_boxes)

def run_test(cfg: dict, device: torch.device):
    """测试集评估：遍历 `images/test`，累积 DetectionMetrics 并保存逐图预测与总体指标。"""
    ckpt_path = cfg["ckpt_path"] or ""
    if not ckpt_path:
        raise SystemExit("未找到权重文件，请在 predict.ckpt_path 指定或训练后再试")
    model = load_model(ckpt_path, device)
    model.eval()
    image_dir = os.path.join(cfg["data_dir"], "images", "test")
    label_dir = os.path.join(cfg["data_dir"], "labels", "test")
    save_base = os.path.join(cfg["save_dir"], "test")
    preds_dir = os.path.join(save_base, "preds")
    preds_raw_dir = os.path.join(save_base, "preds_raw")
    vis_dir = os.path.join(save_base, "vis")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(preds_raw_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.isdir(label_dir):
        tfm = build_transform(cfg["img_size"])
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = []
        if os.path.isdir(image_dir):
            files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in img_exts]
        count = 0
        with torch.no_grad():
            for img_file in sorted(files):
                img_path = os.path.join(image_dir, img_file)
                img = Image.open(img_path).convert("RGB")
                img_t = tfm(img).unsqueeze(0).to(device)
                pred = model(img_t)
                boxes, raw_boxes = decode_boxes(pred, cfg["S"], cfg["conf_thresh"], cfg["iou_thresh"]) 
                base = os.path.splitext(os.path.basename(img_file))[0]
                save_txt(os.path.join(preds_dir, f"{base}.txt"), boxes)
                save_txt(os.path.join(preds_raw_dir, f"{base}.txt"), raw_boxes)
                draw_boxes(img_path, boxes, os.path.join(vis_dir, f"{base}_pred.jpg"))
                count += 1
        print(f"labels 路径不存在，仅预测完成: {count} 张")
        return

    ds = QRCodeDataset(data_dir=cfg["data_dir"], img_size=cfg["img_size"], S=cfg["S"], if_train=False)
    loader = DataLoader(ds, batch_size=cfg.get("test_batch_size", 64), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    metrics = DetectionMetrics(iou_thresh=cfg["iou_thresh"], conf_thresh=cfg["conf_thresh"])
    count = 0

    all_det_boxes = []
    all_gt_boxes = {}

    with torch.no_grad():
        global_index = 0
        for (imgs, targets) in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            pred = model(imgs)
            metrics.update(pred, targets, cfg["S"]) 
            B = pred.size(0)
            for b in range(B):
                boxes, raw_boxes = decode_boxes(pred, cfg["S"], cfg["conf_thresh"], cfg["iou_thresh"], sample_idx=b)
                # Collect predictions for AP calculation (Using RAW boxes)
                for box, score in raw_boxes:
                    all_det_boxes.append((global_index, score, *box))
                # Collect ground truths
                gts = []
                Sg = cfg["S"]
                t = targets[b].cpu() # (S, S, 5)
                for j in range(Sg):
                    for i in range(Sg):
                        if t[j, i, 4] == 1:
                            x_off, y_off, w, h = t[j, i, 0:4].tolist()
                            gts.append(cell_to_bbox(i, j, x_off, y_off, w, h, Sg))
                all_gt_boxes[global_index] = gts

                img_path, _ = ds.samples[global_index]
                base = os.path.splitext(os.path.basename(img_path))[0]
                save_txt(os.path.join(preds_dir, f"{base}.txt"), boxes)
                save_txt(os.path.join(preds_raw_dir, f"{base}.txt"), raw_boxes)
                draw_boxes(img_path, boxes, os.path.join(vis_dir, f"{base}_pred.jpg"))
                global_index += 1
            count = global_index

    precision, recall, miou = metrics.compute()
    ap, precisions, recalls = compute_ap(all_det_boxes, all_gt_boxes, cfg["iou_thresh"])
    print(f"Test Results: Precision={precision:.4f}, Recall={recall:.4f}, mIoU={miou:.4f}, AP={ap:.4f}")
    
    plot_pr_curve(precisions, recalls, ap, os.path.join(save_base, "pr_curve.png"))

    metrics_path = os.path.join(save_base, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"precision": precision, "recall": recall, "mIoU": miou, "AP": ap, "count": count}, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv1_QR 推理与测试评估",
        epilog=(
            "示例:\n"
            "  单图预测:\n"
            "    python predict.py --config ./config.yaml --image ./dataset/images/test/sample.jpg --out ./predictions/single\n"
            "  测试评估:\n"
            "    python predict.py --config ./config.yaml --test\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=str, default="./config.yaml", help="配置文件路径")
    parser.add_argument("--image", type=str, default=None, help="单图预测的图片路径")
    parser.add_argument("--out", type=str, default=None, help="单图预测输出目录，若省略则使用配置中的 out_dir")
    parser.add_argument("--test", action="store_true", help="运行测试集评估（忽略 --image/--out）")
    parser.add_argument("--test-batch-size", type=int, default=None, help="测试评估的批量大小，默认读取配置或 64")
    args = parser.parse_args()
    cfg = read_predict_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.__dict__.get("test_batch_size") is not None:
        cfg["test_batch_size"] = int(args.__dict__.get("test_batch_size"))
    if args.test or cfg["run_test"]:
        run_test(cfg, device)
        return
    image_path = args.image if args.image is not None else (cfg.get("image_path") or None)
    out_dir = args.out if args.out is not None else (cfg.get("out_dir") or None)
    run_single(cfg, image_path, out_dir, device)

if __name__ == "__main__":
    main()
