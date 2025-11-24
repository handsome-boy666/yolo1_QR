from typing import Sequence, Tuple, List

def nms(boxes: List[Tuple[float, float, float, float]], scores: List[float], iou_thresh: float) -> List[int]:
    """对 (boxes, scores) 执行 NMS，返回保留的索引。"""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda k: scores[k], reverse=True)    # 按置信度降序排序
    keep: List[int] = []
    for idx in order:
        b = boxes[idx]
        suppressed = False  # 是否被抑制
        for k in keep:
            if box_iou(b, boxes[k]) > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(idx)
    return keep

def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """计算两个框的交并比（IoU）。

    输入框均为归一化坐标 `(x1, y1, x2, y2)`，范围在 `[0,1]`。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0.0 else 0.0