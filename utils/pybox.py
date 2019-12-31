import torch
import numpy as np
from box import cpu_nms, cpu_overlap


def torch_nms(dets, thresh):
    """
    dets has to be a tensor
    """
    if isinstance(dets, np.ndarray):
        dets = torch.from_numpy(dets).float().contiguous()
        
    if not dets.is_cuda:
        z = dets[:, 1]
        y = dets[:, 2]
        x = dets[:, 3]
        d = dets[:, 4]
        h = dets[:, 5]
        w = dets[:, 6]
        scores = dets[:, 0]

        areas = d * h * w
        order = scores.sort(0, descending=True)[1]
        # order = torch.from_numpy(np.ascontiguousarray(scores.numpy().argsort()[::-1])).long()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        cpu_nms(keep, num_out, dets, order, areas, thresh)

        return dets[keep[:num_out[0]]], keep[:num_out[0]]

    else:
        raise NotImplementedError


def torch_overlap(boxes1, boxes2):
    """
    dets has to be a tensor
    """
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1).float().contiguous()
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2).float().contiguous()

    if not boxes1.is_cuda and not boxes2.is_cuda:
        assert isinstance(boxes1, torch.FloatTensor) and isinstance(boxes2, torch.FloatTensor)
        overlap = torch.zeros([len(boxes1), len(boxes2)], dtype=torch.float32)
        cpu_overlap(boxes1, boxes2, overlap)

        return overlap
    else:
        raise NotImplementedError
