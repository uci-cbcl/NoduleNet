import numpy as np
from net.layer.util import box_transform, box_transform_inv, clip_boxes
import itertools
import torch.nn.functional as F
import torch
from torch.autograd import Variable
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap

def make_rpn_windows(f, cfg):
    """
    Generating anchor boxes at each voxel on the feature map,
    the center of the anchor box on each voxel corresponds to center
    on the original input image.

    return
    windows: list of anchor boxes, [z, y, x, d, h, w]
    """
    stride = cfg['stride']
    anchors = np.asarray(cfg['anchors'])
    offset = (float(stride) - 1) / 2
    _, _, D, H, W = f.shape
    oz = np.arange(offset, offset + stride * (D - 1) + 1, stride)
    oh = np.arange(offset, offset + stride * (H - 1) + 1, stride)
    ow = np.arange(offset, offset + stride * (W - 1) + 1, stride)

    windows = []
    for z, y , x , a in itertools.product(oz, oh , ow , anchors):
        windows.append([z, y , x , a[0], a[1], a[2]])
    windows = np.array(windows)

    return windows

def rpn_nms(cfg, mode, inputs, window, logits_flat, deltas_flat):
    if mode in ['train',]:
        nms_pre_score_threshold = cfg['rpn_train_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rpn_train_nms_overlap_threshold']

    elif mode in ['eval', 'valid', 'test',]:
        nms_pre_score_threshold = cfg['rpn_test_nms_pre_score_threshold']
        nms_overlap_threshold   = cfg['rpn_test_nms_overlap_threshold']

    else:
        raise ValueError('rpn_nms(): invalid mode = %s?'%mode)


    logits = torch.sigmoid(logits_flat).data.cpu().numpy()
    deltas = deltas_flat.data.cpu().numpy()
    batch_size, _, depth, height, width = inputs.size()

    proposals = []
    for b in range(batch_size):
        proposal = [np.empty((0, 8),np.float32),]

        ps = logits[b, : , 0].reshape(-1, 1)
        ds = deltas[b, :, :]

        # Only those anchor boxes larger than a pre-defined threshold
        # will be chosen for nms computation
        index = np.where(ps[:, 0] > nms_pre_score_threshold)[0]
        if len(index) > 0:
            p = ps[index]
            d = ds[index]
            w = window[index]
            box = rpn_decode(w, d, cfg['box_reg_weight'])
            box = clip_boxes(box, inputs.shape[2:])

            output = np.concatenate((p, box),1)

            output = torch.from_numpy(output)
            output, keep = torch_nms(output, nms_overlap_threshold)

            prop = np.zeros((len(output), 8),np.float32)
            prop[:, 0] = b
            prop[:, 1:8] = output
            
            proposal.append(prop)

        proposal = np.vstack(proposal)
        proposals.append(proposal)

    proposals = np.vstack(proposals)

    # Just in case if there is no proposal, we still return a Tensor,
    # torch.from_numpy() cannot take input with 0 dim
    if len(proposals) != 0:
        proposals = Variable(torch.from_numpy(proposals)).cuda()
        return proposals
    else:
        return Variable(torch.rand([0, 8])).cuda()

    return proposals

def rpn_encode(window, truth_box, weight):
    return box_transform(window, truth_box, weight)

def rpn_decode(window, delta, weight):
    return box_transform_inv(window, delta, weight)
