import numpy as np


def box_transform(windows, targets, weight):
    """
    Calculate regression terms, dz, dy, dx, dd, dh, dw
    # windows should equal to # targets
    windows: [num_window, z, y, x, D, H, W]
    targets: [num_target, z, y, x, D, H, W]
    """
    wz, wy, wx, wd, wh, ww = weight
    bz, by, bx = windows[:, 0], windows[:, 1], windows[:, 2]
    bd, bh, bw = windows[:, 3], windows[:, 4], windows[:, 5]
    tz, ty, tx = targets[:, 0], targets[:, 1], targets[:, 2]
    td, th, tw = targets[:, 3], targets[:, 4], targets[:, 5]

    dz = wz * (tz - bz) / bd
    dy = wy * (ty - by) / bh
    dx = wx * (tx - bx) / bw
    dd = wd * np.log(td / bd)
    dh = wh * np.log(th / bh)
    dw = ww * np.log(tw / bw)

    deltas = np.vstack((dz, dy, dx, dd, dh, dw)).transpose()
    return deltas


def box_transform_inv(windows, deltas, weight):
    """
    Apply regression terms to predicted bboxes
    windows: [num_window, z, y, x, D, H, W]
    targets: [num_target, z, y, x, D, H, W]
    """
    num   = len(windows)
    wz, wy, wx, wd, wh, ww = weight
    predictions = np.zeros((num, 6), dtype=np.float32)

    bz, by, bx = windows[:, 0], windows[:, 1], windows[:, 2]
    bd, bh, bw = windows[:, 3], windows[:, 4], windows[:, 5]
    bz = bz[:, np.newaxis]
    by = by[:, np.newaxis]
    bx = bx[:, np.newaxis]
    bd = bd[:, np.newaxis]
    bh = bh[:, np.newaxis]
    bw = bw[:, np.newaxis]

    dz = deltas[:, 0::6] / wz
    dy = deltas[:, 1::6] / wy
    dx = deltas[:, 2::6] / wx
    dd = deltas[:, 3::6] / wd
    dh = deltas[:, 4::6] / wh
    dw = deltas[:, 5::6] / ww

    z = dz * bd + bz
    y = dy * bh + by
    x = dx * bw + bx
    
    d = np.exp(dd) * bd
    h = np.exp(dh) * bh
    w = np.exp(dw) * bw

    predictions[:, 0::6] = z
    predictions[:, 1::6] = y
    predictions[:, 2::6] = x 
    predictions[:, 3::6] = d
    predictions[:, 4::6] = h
    predictions[:, 5::6] = w

    return predictions


def clip_boxes(boxes, img_size):
    """
    clip boxes outside the image, all box follows [p, z, y, x, d, h, w]
    """
    depth, height, width = img_size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, depth  - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width  - 1)

    return boxes