"""通用工具方法"""



def box_iou(box1, box2):
    """
    计算两个框的交并比（IoU）
    :param box1: 第一个框，格式为 (x1, y1, x2, y2)
    :param box2: 第二个框，格式为 (x1, y1, x2, y2)
    :return: 交并比（IoU）
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

    return iou

# 仅保留通用 IoU 函数，预测相关已迁移到 predict.py
