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
    pred = pred.view(batch_size, S, S, 5)

    # 防御非法数值：
    # - 将输入中的 NaN/Inf 转换为有限值，避免在后续 loss 计算中传播导致崩溃
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

    obj_mask = target[..., 4] > 0.5
    noobj_mask = ~obj_mask

    # 坐标损失（仅对有目标网格）
    xy_loss = ((pred[..., 0:2] - target[..., 0:2]) ** 2).sum(dim=-1)
    # 对 w/h 先做下限裁剪后再开方：
    # - 在 0 附近 sqrt 的梯度很大，易引发梯度爆炸/NaN；
    # - clamp 到最小 1e-6 后再 sqrt，可以显著提升稳定性。
    p_wh = torch.sqrt(torch.clamp(pred[..., 2:4], min=1e-6))
    t_wh = torch.sqrt(torch.clamp(target[..., 2:4], min=1e-6))
    wh_loss = ((p_wh - t_wh) ** 2).sum(dim=-1)
    coord_total = (xy_loss + wh_loss)[obj_mask].sum()

    # 置信度损失（分有目标与无目标）
    # 置信度限定在 [0,1] 区间，保持数值合理范围
    conf_pred = torch.clamp(pred[..., 4], 0.0, 1.0)
    obj_conf_loss = ((conf_pred - target[..., 4]) ** 2)[obj_mask].sum()
    noobj_conf_loss = (conf_pred ** 2)[noobj_mask].sum()

    total = lambda_coord * coord_total + obj_conf_loss + lambda_noobj * noobj_conf_loss
    # 总损失做最后一次非有限值清理，避免极端情况下返回 NaN/Inf
    total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)
    return total / batch_size
