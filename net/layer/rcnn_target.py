import copy
from net.layer.rcnn_nms import rcnn_encode
from net.layer.mask_target import make_one_mask_target
import time
import random
import numpy as np
import torch
from torch.autograd import Variable
try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap

score = 1

def add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label, score=1):
    if len(truth_box) !=0:
        truth = np.zeros((len(truth_box), 8),np.float32)
        truth[:, 0] = b
        truth[:, 2:8] = truth_box
        truth[:, 1] = score #1  #
    else:
        truth = np.zeros((0, 8),np.float32)

    sampled_proposal = np.vstack([proposal,truth])
    return sampled_proposal


def make_one_rcnn_target(cfg, input, proposal, truth_box, truth_label):
    sampled_proposal = torch.zeros((0, 8)).float().cuda()
    sampled_label = torch.zeros((0, 1)).long().cuda()
    sampled_assign = np.zeros((0, 1), dtype=np.int32) - 1
    sampled_target = torch.zeros((0, 6)).float().cuda()

    # Even if there is no ground truth box in this batch
    if len(proposal) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    if len(truth_box) == 0:
        num_bg = min(len(proposal), cfg['rcnn_train_batch_size'])
        bg_length = len(proposal)
        bg_index = np.arange(len(proposal))
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)
        ]
        sampled_proposal = proposal[bg_index]
        sampled_proposal = torch.from_numpy(sampled_proposal).cuda()
        sampled_label = torch.zeros((num_bg)).long().cuda()

        return sampled_proposal, sampled_label, sampled_assign, sampled_target 

    _, depth, height, width = input.size()
    num_proposal = len(proposal)
    box = proposal[:, 2:8]

    # Determine positive or negative purely based on threshold
    # Since the GT box has been added to proposal, it is gauranteed that
    # each ground truth would have one proposal
    overlap = torch_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap,1)
    max_overlap = overlap[np.arange(num_proposal),argmax_overlap]

    fg_index = np.where(max_overlap >= cfg['rcnn_train_fg_thresh_low'])[0]
    bg_index = np.where(max_overlap <  cfg['rcnn_train_bg_thresh_high'])[0]

    # sampling for class balance
    num_class = cfg['num_class']
    num = cfg['rcnn_train_batch_size']
    num_fg = int(np.round(cfg['rcnn_train_fg_fraction'] * cfg['rcnn_train_batch_size']))

    fg_length = len(fg_index)
    bg_length = len(bg_index)
    #print(fg_inds_length)

    sampled_assign = argmax_overlap[fg_index]

    # Need to consider four cases, corner cases
    if fg_length > 0 and bg_length > 0:
        idx = []
        idx = random.sample(range(len(fg_index)), min(num_fg, len(fg_index)))
        
        fg_index = fg_index[idx]
        num_fg = len(fg_index)

        num_bg  = num - num_fg
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)
        ]
    elif fg_length > 0:  #no bgs
        idx = []
        idx = random.sample(range(len(fg_index)), min(num_fg, len(fg_index)))
        
        fg_index = fg_index[idx]
        num_fg = len(fg_index)
        num = num_fg
        num_bg = 0

    elif bg_length > 0:  #no fgs
        print('[RCNN] No fgs')
        print(truth_box)
        print('---------------------------')
        print(proposal)

        num_fg = 0
        num_bg = num
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length<num_bg)
        ]
        num_fg_proposal = 0
    else:
        # no bgs and no fgs?
        # raise NotImplementedError
        print('[RCNN] No bgs or fgs')
        print(truth_box)
        print('---------------------------')
        print(proposal)
        num_fg   = 0
        num_bg   = num
        bg_index = np.random.choice(num_proposal, size=num_bg, replace=num_proposal<num_bg)

    assert ((num_fg+num_bg)== num)

    # selecting both fg and bg
    index = np.concatenate([fg_index, bg_index], 0)
    sampled_proposal = np.take(proposal, index, axis=0)

    # label
    sampled_assign = np.take(argmax_overlap, index)
    sampled_label = np.take(truth_label, sampled_assign)
    
    sampled_label[num_fg:] = 0   # Clamp labels for the background to 0
    sampled_assign[num_fg:] = -1 # Clamp label assignments for the background to -1

    # bounding box regression terms
    if num_fg>0:
        target_truth_box = truth_box[sampled_assign[:num_fg], :]
        if len(target_truth_box.shape) < 2: # one dimension lost after slicing
            target_truth_box = target_truth_box[np.newaxis, ...]
        target_box = sampled_proposal[:num_fg,:][:, 2:8]
        sampled_target = rcnn_encode(target_box, target_truth_box, cfg['box_reg_weight'])

    sampled_target   = Variable(torch.from_numpy(sampled_target)).float().cuda()
    sampled_label    = Variable(torch.from_numpy(sampled_label)).long().cuda()
    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()

    return sampled_proposal, sampled_label, sampled_assign, sampled_target


def make_rcnn_target(cfg, mode, inputs, proposals, truth_boxes, truth_labels, truth_masks):
    truth_boxes = copy.deepcopy(truth_boxes)
    truth_labels = copy.deepcopy(truth_labels)
    truth_masks = copy.deepcopy(truth_masks)
    batch_size = len(inputs)
    for b in range(batch_size):
        index = np.where(truth_labels[b] > 0)[0]
        truth_boxes [b] = truth_boxes [b][index]
        truth_labels[b] = truth_labels[b][index]

    proposals = proposals.cpu().data.numpy()
    sampled_proposals = []
    sampled_labels = []
    sampled_assigns = []
    sampled_targets = []
    sampled_masks = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input = inputs[b]
        truth_box = truth_boxes[b]
        truth_label = truth_labels[b]

        if len(proposals) == 0:
            proposal = np.zeros((0, 8),np.float32)
        else:
            proposal = proposals[proposals[:,0] == b]

        # Add ground truth box to proposal, so that even if the RPN branch fails to find something,
        # we can still get classification branch to work
        proposal = add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label)

        sampled_proposal, sampled_label, sampled_assign, sampled_target = \
           make_one_rcnn_target(cfg, input, proposal, truth_box, truth_label)

        sampled_proposals.append(sampled_proposal)
        sampled_labels.append(sampled_label)
        sampled_assigns.append(sampled_assign)
        sampled_targets.append(sampled_target)

    sampled_proposals = torch.cat(sampled_proposals, 0)
    sampled_labels    = torch.cat(sampled_labels, 0)
    sampled_targets   = torch.cat(sampled_targets, 0)
    sampled_assigns   = np.hstack(sampled_assigns)

    return sampled_proposals, sampled_labels, sampled_assigns, sampled_targets


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

 
