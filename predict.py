import os
import json
import math
import argparse
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

from models.yolov1QR import YOLOv1_QR
from utils.utils import box_iou


def draw_boxes(image_path, boxes, scores, output_path, color='red', width=3):
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


def _decode_yolo_grid(pred_grid, orig_w, orig_h, S, conf_thresh=0.3):
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


def _nms(boxes, scores, iou_thresh=0.5):
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


def predict_image(image_path, checkpoint_path, img_size=512, S=4, conf_thresh=0.3, nms_thresh=0.5, output_path=None, device=None, save_json=True):
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


def main():

    """
    python predict.py --image "图片路径" --checkpoint "epoch_048.pth" --img-size 512 --S 4 --conf-thresh 0.2 --nms-thresh 0.5 --device cuda
    """
    parser = argparse.ArgumentParser(description='YOLOv1_QR 单图预测')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--checkpoint', required=True, help='模型权重 .pth 路径')
    parser.add_argument('--img-size', type=int, default=512, help='输入尺寸，需与训练一致')
    parser.add_argument('--S', type=int, default=4, help='网格大小，默认4，若模型输出不一致自动覆盖')
    parser.add_argument('--conf-thresh', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--nms-thresh', type=float, default=0.5, help='NMS IoU 阈值')
    parser.add_argument('--output', default=None, help='输出可视化图片路径')
    parser.add_argument('--device', default=None, choices=[None, 'cpu', 'cuda'], help='设备选择')
    parser.add_argument('--no-json', action='store_true', help='不保存 JSON 结果')
    args = parser.parse_args()

    device = None
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_json = not args.no_json

    boxes, scores, out = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        S=args.S,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        output_path=args.output,
        device=device,
        save_json=save_json,
    )

    print(f'[Predict] 保存可视化到: {out}')
    print(f'[Predict] boxes: {boxes}')
    print(f'[Predict] scores: {scores}')


if __name__ == '__main__':
    main()