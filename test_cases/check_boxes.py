import os
import sys
sys.path.append('../')

import torch
import numpy as np
from utils.pybox import torch_nms, torch_overlap
from utils.util import py_box_overlap, py_nms
import time


def check():
    boxes1 = torch.FloatTensor([[5, 5, 5, 10, 10, 10],
                                [10, 10, 20, 6, 7, 8],
                                [15, 10, 5, 10, 11, 11]
                                ])
    boxes2 = torch.FloatTensor([[5, 5, 5, 10, 10, 10],
                                [10, 10, 20, 6, 7, 8],
                                [15, 10, 5, 10, 11, 11]
                                ])

    nms_boxes = torch.FloatTensor([[0.99, 10, 10, 20, 7, 8, 9],
                                [0.9, 10, 10, 20, 6, 7, 8],
                                [0.5, 15, 10, 5, 10, 11, 11]
                                ])

    s = time.time()
    print('=============py_box_overlap=============')
    print py_box_overlap(boxes1, boxes2)
    print('=============py_box_overlap: %.5f=============' % (time.time() - s))

    s = time.time()
    print('=============torch_box_overlap=============')
    print torch_overlap(boxes1, boxes2)
    print('=============torch_box_overlap: %.5f=============' % (time.time() - s))

    s = time.time()
    print('=============py_nms=============')
    print py_nms(nms_boxes.numpy(), 0.5)
    print('=============py_nms: %.5f=============' % (time.time() - s))

    s = time.time()
    print('=============torch_nms=============')
    print torch_nms(nms_boxes, 0.5)
    print('=============torch_nms: %.5f=============' % (time.time() - s))

if __name__ == '__main__':
    check()