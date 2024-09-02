# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import math


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def ciou(bboxes1, bboxes2):
    bboxes1 = torch.sigmoid(bboxes1)
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])
    h1 = torch.exp(bboxes1[:, 3])
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1.unsqueeze(1) - w1.unsqueeze(1) / 2, center_x2.unsqueeze(0) - w2.unsqueeze(0) / 2)
    inter_r = torch.min(center_x1.unsqueeze(1) + w1.unsqueeze(1) / 2, center_x2.unsqueeze(0) + w2.unsqueeze(0) / 2)
    inter_t = torch.max(center_y1.unsqueeze(1) - h1.unsqueeze(1) / 2, center_y2.unsqueeze(0) - h2.unsqueeze(0) / 2)
    inter_b = torch.min(center_y1.unsqueeze(1) + h1.unsqueeze(1) / 2, center_y2.unsqueeze(0) + h2.unsqueeze(0) / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1.unsqueeze(1) - w1.unsqueeze(1) / 2, center_x2.unsqueeze(0) - w2.unsqueeze(0) / 2)
    c_r = torch.max(center_x1.unsqueeze(1) + w1.unsqueeze(1) / 2, center_x2.unsqueeze(0) + w2.unsqueeze(0) / 2)
    c_t = torch.min(center_y1.unsqueeze(1) - h1.unsqueeze(1) / 2, center_y2.unsqueeze(0) - h2.unsqueeze(0) / 2)
    c_b = torch.max(center_y1.unsqueeze(1) + h1.unsqueeze(1) / 2, center_y2.unsqueeze(0) + h2.unsqueeze(0) / 2)

    inter_diag = (center_x2.unsqueeze(0) - center_x1.unsqueeze(1)) ** 2 + (center_y2.unsqueeze(0) - center_y1.unsqueeze(1)) ** 2
    c_diag = torch.clamp((c_r - c_l), min=0) ** 2 + torch.clamp((c_b - c_t), min=0) ** 2

    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2.unsqueeze(0) / h2.unsqueeze(0)) - torch.atan(w1.unsqueeze(1) / h1.unsqueeze(1))), 2)
    with torch.no_grad():
        S = (iou > 0.5).float()
        alpha = S * v / (1 - iou + v)
    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return 1 - cious  # 转换为非负损失值
