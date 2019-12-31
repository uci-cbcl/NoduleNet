from net.lib.box.overlap.torch_overlap.extension import *

import torch
import numpy as np

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
  # elif boxes1.is_cuda and boxes2.is_cuda:
  #   overlap = torch.cuda.FloatTensor(len(boxes1), len(boxes2)).fill_(0)
  #   # overlap = torch.zeros([len(boxes1), len(boxes2)], dtype=torch.float32).cuda()
  #   gpu_overlap(boxes1, boxes2, overlap)
  #   return overlap
  #   # print('[torch_overlap] does not support gpu right now.')
  #   # raise NotImplementedError
  # else:
  #   print('[Error] Two boxes are not on the same device.')
  #   raise NotImplementedError



