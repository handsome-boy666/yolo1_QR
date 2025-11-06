"""通用工具方法

仅包含与项目无关的基础工具函数，保持精简。
"""
from typing import Sequence



def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """计算两个框的交并比（IoU）。

    参数：
    - box1: `(x1, y1, x2, y2)`
    - box2: `(x1, y1, x2, y2)`

    返回：
    - IoU 浮点数（0~1）
    """
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集区域的面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算并集区域的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # 计算交并比（IoU）
    iou = inter_area / union_area if union_area > 0 else 0.0

    return float(iou)

# 仅保留通用 IoU 函数，预测相关已迁移到 predict.py
