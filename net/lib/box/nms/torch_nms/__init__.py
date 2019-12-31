from net.lib.box.nms.torch_nms.extension import *

import torch
import numpy as np

def torch_nms(dets, thresh):
    """
    dets has to be a tensor
    """
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
        z = dets[:, 1]
        y = dets[:, 2]
        x = dets[:, 3]
        d = dets[:, 4]
        scores = dets[:, 0]

        areas = d ** 3
        order = scores.sort(0, descending=True)[1]
        # order = torch.from_numpy(np.ascontiguousarray(scores.cpu().numpy().argsort()[::-1])).long().cuda()

        dets = dets[order].contiguous()

        keep = torch.LongTensor(dets.size(0))
        num_out = torch.LongTensor(1)
        # keep = torch.cuda.LongTensor(dets.size(0))
        # num_out = torch.cuda.LongTensor(1)
        gpu_nms(keep, num_out, dets, thresh)

        # return dets[keep[:num_out[0]].cuda()].contiguous(), keep[:num_out[0]].cuda()
        return dets[keep[:num_out[0]].cuda()].contiguous(), order[keep[:num_out[0]]].contiguous()

