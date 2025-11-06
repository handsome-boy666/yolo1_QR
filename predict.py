"""单图预测脚本与函数

封装 `predict_image` 以便在代码或命令行中使用：
- 加载 `YOLOv1_QR` 权重；
- 预处理到指定尺寸；
- 解析网格输出为像素坐标；
- 应用简易 NMS；
- 保存可视化图片与 JSON 结果（满足测试结果保存要求）。
"""

import os
import json
import math
from typing import List, Tuple, Optional
import yaml
from dataclasses import dataclass

import torch
from torch import Tensor
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

from models.yolov1QR import YOLOv1_QR
from utils.utils import box_iou


def draw_boxes(image_path: str, boxes: List[Tuple[float, float, float, float]], scores: List[float], output_path: str, color: str = 'red', width: int = 3) -> Image.Image:
    """
    在图像上绘制多个框并保存。
    :param image_path: 输入图像路径
    :param boxes: [(x1, y1, x2, y2), ...] 像素坐标
    :param scores: [score, ...]
    :param output_path: 输出图片路径
    :param color: 框颜色
    :param width: 框线宽
    :return: PIL.Image 对象
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(0, y1 - 12)), f"{s:.2f}", fill=color)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return img


def _decode_yolo_grid(pred_grid: Tensor, orig_w: int, orig_h: int, S: int, conf_thresh: float = 0.3) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    """
    将模型输出的网格 (S,S,5) 解析为像素级别的框列表。
    pred_grid: Tensor[S,S,5] -> [x_off, y_off, w, h, conf]
    返回: boxes(List[Tuple]), scores(List[float])
    """
    boxes = []
    scores = []
    for j in range(S):
        for i in range(S):
            x_off, y_off, w_rel, h_rel, conf = pred_grid[j, i].tolist()
            if conf < conf_thresh:
                continue
            # 归一化中心坐标
            cx_rel = (i + max(0.0, min(1.0, x_off))) / S
            cy_rel = (j + max(0.0, min(1.0, y_off))) / S
            # 宽高转像素
            w = max(1e-6, min(1.0, w_rel)) * orig_w
            h = max(1e-6, min(1.0, h_rel)) * orig_h
            cx = cx_rel * orig_w
            cy = cy_rel * orig_h
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(orig_w - 1.0, cx + w / 2)
            y2 = min(orig_h - 1.0, cy + h / 2)
            boxes.append((x1, y1, x2, y2))
            scores.append(float(conf))
    return boxes, scores


def _nms(boxes: List[Tuple[float, float, float, float]], scores: List[float], iou_thresh: float = 0.5) -> List[int]:
    """简易 NMS，返回保留的索引列表。"""
    if not boxes:
        return []
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda k: scores[k], reverse=True)
    keep = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if box_iou(boxes[cur], boxes[i]) < iou_thresh]
    return keep


def predict_image(image_path: str, checkpoint_path: str, img_size: int = 512, S: int = 4, conf_thresh: float = 0.3, nms_thresh: float = 0.5, output_path: Optional[str] = None, device: Optional[torch.device] = None, save_json: bool = True) -> Tuple[List[Tuple[float, float, float, float]], List[float], str]:
    """
    预测并绘制单张图片。
    - 加载指定 checkpoint
    - 前处理到 img_size
    - 解析网格输出为像素框，应用 NMS
    - 保存绘制结果和 JSON 框信息（可选）

    返回: boxes(List[Tuple]), scores(List[float]), output_path(str)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_path}")

    # 读取原图尺寸
    img_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img_orig.size

    # 前处理（与训练一致）
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img_tensor = tfm(img_orig).unsqueeze(0).to(device)  # [1,3,H,W]

    # 模型与权重
    model = YOLOv1_QR().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state'])
    model.eval()

    with torch.no_grad():
        pred = model(img_tensor)  # [1, S*S*5]
        num_feat = pred.size(1)
        S_calc = int(math.sqrt(max(1, num_feat // 5)))
        if S_calc != S:
            S = S_calc  # 以模型输出为准
        pred_grid = pred.view(1, S, S, 5)[0].cpu()
        boxes, scores = _decode_yolo_grid(pred_grid, orig_w, orig_h, S, conf_thresh)

    # NMS
    keep = _nms(boxes, scores, iou_thresh=nms_thresh)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]

    # 输出路径
    if output_path is None:
        base, ext = os.path.splitext(os.path.basename(image_path))
        out_dir = os.path.join(os.path.dirname(image_path), 'predictions')
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"{base}_pred{ext}")

    # 绘制并保存
    draw_boxes(image_path, boxes, scores, output_path)

    # 保存 JSON 结果，便于后续查看
    if save_json:
        meta = {
            'image': os.path.abspath(image_path),
            'checkpoint': os.path.abspath(checkpoint_path),
            'boxes': boxes,
            'scores': scores,
            'S': S,
            'conf_thresh': conf_thresh,
            'nms_thresh': nms_thresh,
        }
        json_path = os.path.splitext(output_path)[0] + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return boxes, scores, output_path


def _load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not files:
        return None
    files.sort()
    return os.path.join(ckpt_dir, files[-1])


def _select_device_from_cfg(device_cfg: str | None) -> torch.device:
    if device_cfg is None or device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_cfg in ('cuda', 'cpu'):
        return torch.device(device_cfg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class PredictConfig:
    """预测运行所需配置（完全由 config.yaml 提供，无需命令行）。"""
    image_path: str
    output_path: Optional[str]
    save_json: bool
    checkpoint_path: str
    img_size: int
    S: int
    conf_thresh: float
    nms_thresh: float
    device: torch.device


def read_predict_config(path: str = './config.yaml') -> PredictConfig:
    """读取并解析预测配置，并确定最终的 checkpoint 与设备。"""
    cfg = _load_config(path)
    pcfg = cfg.get('predict', {})
    image_path: Optional[str] = pcfg.get('image_path')
    output_path: Optional[str] = pcfg.get('output_path')
    save_json: bool = bool(pcfg.get('save_json', True))
    checkpoint_path: Optional[str] = pcfg.get('checkpoint_path')
    ckpt_dir: str = pcfg.get('ckpt_dir', './checkpoints')
    use_latest: bool = bool(pcfg.get('use_latest', True))
    img_size: int = int(pcfg.get('img_size', 512))
    S: int = int(pcfg.get('S', 4))
    conf_thresh: float = float(pcfg.get('conf_thresh', 0.3))
    nms_thresh: float = float(pcfg.get('nms_thresh', 0.5))
    device = _select_device_from_cfg(pcfg.get('device', 'auto'))

    if not image_path:
        raise FileNotFoundError('未指定预测图片路径：请在 config.yaml 的 predict.image_path 填写图片路径。')

    if not checkpoint_path:
        if use_latest:
            checkpoint_path = _find_latest_checkpoint(ckpt_dir)
        if not checkpoint_path:
            raise FileNotFoundError('未找到可用模型：请在 config.yaml 的 predict.checkpoint_path 指定，或开启 use_latest 并确保 ckpt_dir 中存在模型。')

    return PredictConfig(
        image_path=image_path,
        output_path=output_path,
        save_json=save_json,
        checkpoint_path=checkpoint_path,
        img_size=img_size,
        S=S,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
        device=device,
    )


def main() -> None:
    pc = read_predict_config('./config.yaml')

    boxes, scores, out = predict_image(
        image_path=pc.image_path,
        checkpoint_path=pc.checkpoint_path,
        img_size=pc.img_size,
        S=pc.S,
        conf_thresh=pc.conf_thresh,
        nms_thresh=pc.nms_thresh,
        output_path=pc.output_path,
        device=pc.device,
        save_json=pc.save_json,
    )

    print(f'[Predict] 保存可视化到: {out}')
    print(f'[Predict] boxes: {boxes}')
    print(f'[Predict] scores: {scores}')


if __name__ == '__main__':
    main()