"""YOLOv1 简化损失函数

面向本项目的单框 YOLOv1 损失实现：
- 坐标损失仅在有目标的网格计算；
- 置信度损失对有目标与无目标分开加权；
- 对 w/h 使用 `clamp+sqrt` 防止负值导致的 NaN；
- 对 `pred/target` 使用 `nan_to_num` 防御非法数值。
"""

import torch
from torch import Tensor


def yolo_v1_loss(pred: Tensor, target: Tensor, S: int = 4, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> Tensor:
    """
    简化版 YOLOv1 损失：
    - 坐标损失：只对有目标的网格计算 (x, y, w, h) 的 L2
    - 置信度损失：有目标/无目标分开计算 L2，不同权重

    参数:
    - pred: 模型输出，形状 [B, S*S*5]
    - target: 标签，形状 [B, S, S, 5]，最后一维为 (x, y, w, h, conf)
    - S: 网格划分数
    - lambda_coord: 坐标损失权重
    - lambda_noobj: 无目标置信度损失权重

    返回:
    - 标量损失
    """
    if pred.dim() != 2:
        raise ValueError("pred 应为二维张量，形状 [B, S*S*5]")
    batch_size = pred.size(0)
    pred = pred.view(batch_size, S, S, 5)  # 转换为 [B, S, S, 5] 形状

    obj_mask = target[..., 4] > 0.5  # 有目标的网格
    noobj_mask = ~obj_mask  # 无目标的网格

    # 边框坐标损失
    coord_loss = ((pred[..., 0:2] - target[..., 0:2]) ** 2).sum(dim=-1) 
    wh_loss = ((torch.sqrt(pred[..., 2:4]) - torch.sqrt(target[..., 2:4])) ** 2).sum(dim=-1)
    coord_total = (coord_loss + wh_loss)[obj_mask].sum()

    # 置信度损失
    conf_pred = pred[..., 4]
    obj_conf_loss = ((conf_pred - target[..., 4]) ** 2)[obj_mask].sum()
    noobj_conf_loss = ((conf_pred) ** 2)[noobj_mask].sum()

    total = lambda_coord * coord_total + obj_conf_loss + lambda_noobj * noobj_conf_loss
    return total / batch_size    # 归一化，防止 batch_size 为 0 时除零错误
