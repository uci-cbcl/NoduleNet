from . import *
import torch
import numpy as np
import torch.nn.functional as F

def mask_nms(cfg, mode, mask_logits, crop_boxes, inputs):
    nms_overlap_threshold   = cfg['mask_test_nms_overlap_threshold']
    batch_size, _, depth, height, width = inputs.size() #original image width
    num_class = cfg['num_class']
    keep_ids = []

    for b in range(batch_size):
        crop_boxes_batch = crop_boxes[crop_boxes[:, 0] == b]
        crop_boxes_batch = crop_boxes_batch[:, 1:]
        n = len(crop_boxes_batch)
        cur = 0
        visited = [False for _ in range(n)]
        while cur < n:
            if visited[cur]:
                cur += 1
                continue

            visited[cur] = True
            keep_ids.append(cur)
            mask1 = mask_logits2probs(mask_logits[cur])
            for i in range(cur + 1, n):
                mask2 = mask_logits2probs(mask_logits[i])
                if mask_iou(mask1, mask2) > nms_overlap_threshold:
                    visited[i] = True
            
            cur += 1

    return keep_ids

def mask_iou(mask1, mask2):
    return float(np.logical_and(mask1, mask2).sum()) / np.logical_or(mask1, mask2).sum()

def mask_logits2probs(mask):
    mask = (F.sigmoid(mask) > 0.5).cpu().numpy().astype(np.uint8)

    return mask