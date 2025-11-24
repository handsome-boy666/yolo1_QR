"""通用工具方法

包含训练中常用的度量与几何工具，尽量保持简单易懂。
"""
from typing import Sequence, Tuple, List
import torch
from models.utils import box_iou, nms

def cell_to_bbox(i: int, j: int, x_off: float, y_off: float, w: float, h: float, S: int) -> Tuple[float, float, float, float]:
    """将网格内偏移+尺寸转换为整图归一化框。

    - `(i, j)`: 网格坐标（列、行）
    - `(x_off, y_off)`: 该网格内中心偏移（0~1）
    - `(w, h)`: 框的归一化宽高（0~1）
    - `S`: 网格大小
    返回 `(x1, y1, x2, y2)`（均为归一化坐标）
    """
    cx = (i + float(x_off)) / S
    cy = (j + float(y_off)) / S
    x1 = max(0.0, cx - float(w) * 0.5)
    y1 = max(0.0, cy - float(h) * 0.5)
    x2 = min(1.0, cx + float(w) * 0.5)
    y2 = min(1.0, cy + float(h) * 0.5)
    return x1, y1, x2, y2

class DetectionMetrics:
    """简单检测指标聚合器。

    - `precision@iou_thresh`: 基于 IoU 阈值的精度
    - `recall@iou_thresh`: 基于 IoU 阈值的召回
    - `mIoU`: 在匹配到的目标上的 IoU 均值

    预测来源：所有网格中置信度 ≥ `conf_thresh` 的框；
    匹配策略：先对预测执行 NMS 去重，再按 IoU 阈值判定命中并避免重复匹配。
    """

    def __init__(self, iou_thresh: float = 0.5, conf_thresh: float = 0.5) -> None:
        self.iou_thresh = float(iou_thresh)
        self.conf_thresh = float(conf_thresh)
        self.tp = 0  # true positives
        self.fp = 0  # false positives
        self.fn = 0  # false negatives
        self.iou_sum = 0.0  # 累计 IoU
        self.iou_cnt = 0    # 参与平均的样本数

    def reset(self) -> None:
        """清空累计状态。"""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.iou_sum = 0.0
        self.iou_cnt = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, S: int, nms_iou_thresh: float = 0.5) -> None:
        """用一个批次的预测与标签更新累计指标。

        - `preds`: `[B, S*S*5]`，最后维度为 `(x_off, y_off, w, h, conf)`
        - `targets`: `[B, S, S, 5]`，同样是 `(x_off, y_off, w, h, conf)`
        - `S`: 网格大小
        """
        if preds.dim() != 2:
            return

        B = preds.size(0)
        preds = preds.view(B, S, S, 5).detach().cpu()
        targets = targets.detach().cpu()

        for b in range(B):
            # 收集 GT 框（标签中 conf>0.5 视为存在目标）
            gt_boxes: list[Tuple[float, float, float, float]] = []
            for j in range(S):
                for i in range(S):
                    if float(targets[b, j, i, 4]) > 0.5:
                        gt_boxes.append(
                            cell_to_bbox(i, j,
                                         float(targets[b, j, i, 0]),
                                         float(targets[b, j, i, 1]),
                                         float(targets[b, j, i, 2]),
                                         float(targets[b, j, i, 3]), S)
                        )

            # 收集预测框并执行 NMS 过滤
            boxes: List[Tuple[float, float, float, float]] = []
            scores: List[float] = []
            for j in range(S):
                for i in range(S):
                    conf = float(preds[b, j, i, 4])
                    if conf >= self.conf_thresh:
                        boxes.append(cell_to_bbox(
                            i, j,
                            float(preds[b, j, i, 0]),
                            float(preds[b, j, i, 1]),
                            float(preds[b, j, i, 2]),
                            float(preds[b, j, i, 3]), S))
                        scores.append(conf)
            keep_indices = nms(boxes, scores, nms_iou_thresh)   # 对预测框执行 NMS 去重
            pred_boxes: List[Tuple[Tuple[float, float, float, float], float]] = [(boxes[k], scores[k]) for k in keep_indices]

            # 贪心匹配：每个 GT 与未匹配的预测中 IoU 最大者进行比对
            matched: set[int] = set()
            for gt in gt_boxes:
                best_idx = -1
                best_iou = 0.0
                for idx, (pb, _) in enumerate(pred_boxes):
                    if idx in matched:
                        continue
                    iou = box_iou(gt, pb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= self.iou_thresh:   # 仅当 IoU 超过阈值时才算匹配
                    matched.add(best_idx)
                    self.tp += 1
                    self.iou_sum += best_iou
                    self.iou_cnt += 1
                else:
                    self.fn += 1

            # 剩余未匹配的预测视为 FP
            self.fp += max(0, len(pred_boxes) - len(matched))

    def compute(self) -> Tuple[float, float, float]:
        """返回 `(precision, recall, mIoU)`。"""
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        miou = self.iou_sum / self.iou_cnt if self.iou_cnt > 0 else 0.0
        return precision, recall, miou

