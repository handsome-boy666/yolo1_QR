import torch


def yolo_v1_loss(pred, target, S=4, lambda_coord=5.0, lambda_noobj=0.5):
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
    B = pred.size(0)
    pred = pred.view(B, S, S, 5)
    # 防御非法数值，避免 NaN/Inf 传播
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

    obj_mask = target[..., 4] > 0.5
    noobj_mask = ~obj_mask

    # 坐标与尺寸损失，仅对有目标的网格
    coord_loss = ((pred[..., 0:2] - target[..., 0:2]) ** 2).sum(dim=-1)
    # 对 w/h 使用 clamp+sqrt，避免负值导致 sqrt NaN
    pred_wh = torch.clamp(pred[..., 2:4], min=1e-6)
    target_wh = target[..., 2:4]  # 不对 target_wh 做 clamp
    wh_loss = ((torch.sqrt(pred_wh) - torch.sqrt(target_wh)) ** 2).sum(dim=-1)
    coord_total = (coord_loss + wh_loss)[obj_mask].sum()

    # 置信度损失（对预测再做一次防御）
    conf_pred = torch.nan_to_num(pred[..., 4], nan=0.0, posinf=1e4, neginf=-1e4)
    obj_conf_loss = ((conf_pred - target[..., 4]) ** 2)[obj_mask].sum()
    noobj_conf_loss = ((conf_pred) ** 2)[noobj_mask].sum()

    total = lambda_coord * coord_total + obj_conf_loss + lambda_noobj * noobj_conf_loss
    return total / max(B, 1)    # 归一化，防止 batch_size 为 0 时除零错误