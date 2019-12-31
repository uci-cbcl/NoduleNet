from . import *
import torch
import numpy as np
from scipy.ndimage import zoom

def make_one_mask_target(cfg, mode, input, sampled_proposal, sampled_assign, truth_box, truth_mask):
    """
    Deprecated.

    Was used for generating mask for MaskRcnn
    """
    sampled_mask = []
    mask_crop_size = cfg['mask_crop_size']
    for i in range(len(sampled_proposal)):
        _, D, H, W = input.shape
        target_id = sampled_assign[i]

        # If this is a negative proposal
        if target_id < 0:
            continue

        box = sampled_proposal[i, 2:8]
        mask = truth_mask[target_id]

        z, y, x, d, h, w = box
        z_start = max(0, int(np.floor(z - d / 2.)))
        y_start = max(0, int(np.floor(y - h / 2.)))
        x_start = max(0, int(np.floor(x - w / 2.)))
        z_end = min(D, int(np.ceil(z + d / 2.)))
        y_end = min(H, int(np.ceil(y + h / 2.)))
        x_end = min(W, int(np.ceil(x + w / 2.)))

        crop = mask[z_start:z_end, y_start:y_end, x_start:x_end]
        D_c, H_c, W_c = crop.shape
        zoomed_crop = zoom(crop, 
                          (float(mask_crop_size[0]) / D_c, float(mask_crop_size[1]) / H_c, float(mask_crop_size[2]) / W_c), 
                          order=2)
        zoomed_crop = (zoomed_crop > 0.5).astype(np.float32)
        sampled_mask.append(zoomed_crop)

    sampled_mask = np.array(sampled_mask)
    sampled_mask = torch.from_numpy(sampled_mask).cuda()

    return sampled_mask


def make_mask_target(cfg, mode, inputs, crop_boxes, truth_boxes, truth_labels, masks):
    target_masks = []
    for detection in crop_boxes:
        b, z_start, y_start, x_start, z_end, y_end, x_end, cat = detection
        mask = torch.from_numpy(masks[b][cat - 1][z_start:z_end, y_start:y_end, x_start:x_end]).cuda()
        target_masks.append(mask)

    return target_masks