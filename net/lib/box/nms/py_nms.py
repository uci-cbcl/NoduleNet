import numpy as np

# deafult nms in python for checking
def py_nms(dets, thresh):
    z = dets[:, 1]
    y = dets[:, 2]
    x = dets[:, 3]
    d = dets[:, 4]
    h = dets[:, 5]
    w = dets[:, 6]
    scores = dets[:, 0]

    areas = d * h * w
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx0 = np.maximum(x[i] - w[i] / 2, x[order[1:]] - w[order[1:]] / 2)
        yy0 = np.maximum(y[i] - h[i] / 2, y[order[1:]] - h[order[1:]] / 2)
        zz0 = np.maximum(z[i] - d[i] / 2, z[order[1:]] - d[order[1:]] / 2)
        xx1 = np.minimum(x[i] + w[i] / 2, x[order[1:]] + w[order[1:]] / 2)
        yy1 = np.minimum(y[i] + h[i] / 2, y[order[1:]] + h[order[1:]] / 2)
        zz1 = np.minimum(z[i] + d[i] / 2, z[order[1:]] + d[order[1:]] / 2)

        inter_w = np.maximum(0.0, xx1 - xx0)
        inter_h = np.maximum(0.0, yy1 - yy0)
        inter_d = np.maximum(0.0, zz1 - zz0)
        intersect = inter_w * inter_h * inter_d
        overlap = intersect / (areas[i] + areas[order[1:]] - intersect)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]

    return dets[keep], keep