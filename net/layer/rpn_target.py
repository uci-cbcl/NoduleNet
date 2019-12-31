try:
    from utils.pybox import *
except ImportError:
    print('Warning: C++ module import failed! This should only happen in deployment')
    from utils.util import py_nms as torch_nms
    from utils.util import py_box_overlap as torch_overlap
import random
from net.layer.rpn_nms import rpn_encode
from torch.autograd import Variable


def make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label):
    """
    Generate region proposal targets for one batch

    cfg: dict, for hyper-parameters
    mode: string, which phase/mode is used currently
    input: 5D torch tensor of [batch, channel, z, y, x], original input to the network
    window: list of anchor bounding boxes, [z, y, x, d, h, w]
    truth_box: list of ground truth bounding boxes, [z, y, x, d, h, w]
    truth_label: list of grount truth class label for each object in the correponding truth_box

    return torch tensors
    label: positive or negative (1 or 0) for each anchor box
    label_assign: index of the ground truth box, to which the anchor box is matched to
    label_weight: class weight for each sample, zero means current sample is protected,
                  and won't contribute to loss
    target: bounding box regression terms
    target_weight: weight for each regression term, by default it should all be ones
    """

    num_neg = cfg['num_neg']
    num_window = len(window)
    label = np.zeros((num_window, ), np.float32)
    label_assign = np.zeros((num_window, ), np.int32) - 1
    label_weight = np.zeros((num_window, ), np.float32)
    target = np.zeros((num_window, 6), np.float32)
    target_weight = np.zeros((num_window, ), np.float32)


    num_truth_box = len(truth_box)
    if num_truth_box:

        _, depth, height, width = input.size()

        # Get sure background anchor boxes
        overlap = torch_overlap(window, truth_box)

        # For each anchor box, get the index of the ground truth box that
        # has the largest IoU with it
        argmax_overlap = np.argmax(overlap, 1)

        # For each anchor box, get the IoU of the ground truth box that
        # has the largest IoU with it
        max_overlap = overlap[np.arange(num_window), argmax_overlap]

        # The anchor box is a sure background, if its largest IoU is less than
        # a threshold
        bg_index = np.where(max_overlap < cfg['rpn_train_bg_thresh_high'])[0]
        label[bg_index] = 0
        label_weight[bg_index] = 1

        # The anchor box is a sure foreground, if its largest IoU is larger or 
        # equal than a threshold
        fg_index = np.where(max_overlap >= cfg['rpn_train_fg_thresh_low'])[0]
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[...] = argmax_overlap

        # In case no anchor box that overlaps with the ground truth box meets the threshold, 
        # for each ground truth box, we include anchor box that has the highest IoU with it, 
        # include multiple maxs if there exists more than one anchor box
        argmax_overlap = np.argmax(overlap,0)
        max_overlap = overlap[argmax_overlap,np.arange(num_truth_box)]
        argmax_overlap, a = np.where(overlap == max_overlap)

        fg_index = argmax_overlap

        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[fg_index] = a

        # In case one ground truth box within one batch has way too many positive anchors, 
        # which may affect the sample in the loss fucntion,
        # we only random one positive anchor for each ground truth box
        fg_index = np.where(label != 0)[0]
        idx = random.sample(range(len(fg_index)), 1)
        label[fg_index] = 0
        label_weight[fg_index] = 0
        fg_index = fg_index[idx]
        label[fg_index] = 1
        label_weight[fg_index] = 1


        # Prepare regression terms for each positive anchor
        fg_index = np.where(label != 0)[0]
        target_window = window[fg_index]
        target_truth_box = truth_box[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box, cfg['box_reg_weight'])
        target_weight[fg_index] = 1

        # This should by no means be used, left just in case
        invalid_truth_label = np.where(truth_label < 0)[0]
        invalid_index = np.isin(label_assign, invalid_truth_label) & (label!=0)
        label_weight[invalid_index] = 0
        target_weight[invalid_index] = 0

        if mode in ['train']:
            fg_index = np.where( (label_weight!=0) & (label!=0))[0]
            bg_index = np.where( (label_weight!=0) & (label==0))[0]

            # Random sample num_neg negative anchor boxes first
            # This is very strange, but it works well in practice
            # It makes the use of hard negative example mining loss, not 
            # actually hard negative example mining.
            label_weight[bg_index] = 0
            idx = random.sample(range(len(bg_index)), min(num_neg, len(bg_index)))
            bg_index = bg_index[idx]

            # Calculate weight for class balance
            num_fg = max(1, len(fg_index))
            num_bg = len(bg_index)
            label_weight[bg_index] = float(num_fg)/num_bg

        target_weight[fg_index] = label_weight[fg_index]
    else:
        # if there is no ground truth box in this batch
        label_weight[...] = 1

        if mode in ['train']:
            bg_index = np.where((label_weight!=0) & (label==0))[0]

            label_weight[bg_index] = 0
            idx = random.sample(range(len(bg_index)), min(num_neg, len(bg_index)))
            bg_index = bg_index[idx]
            label_weight[bg_index] = 1.0 / len(bg_index)


    label = Variable(torch.from_numpy(label)).cuda()
    label_assign = Variable(torch.from_numpy(label_assign)).cuda()
    label_weight = Variable(torch.from_numpy(label_weight)).cuda()
    target = Variable(torch.from_numpy(target)).cuda()
    target_weight = Variable(torch.from_numpy(target_weight)).cuda()
    return  label, label_assign, label_weight, target, target_weight


def make_rpn_target(cfg, mode, inputs, window, truth_boxes, truth_labels):
    rpn_labels = []
    rpn_label_assigns = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []

    batch_size = len(inputs)
    for b in range(batch_size):
        input = inputs[b]
        truth_box   = truth_boxes[b]
        truth_label = truth_labels[b]

        rpn_label, rpn_label_assign, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label)

        rpn_labels.append(rpn_label.view(1, -1))
        rpn_label_assigns.append(rpn_label_assign.view(1, -1))
        rpn_label_weights.append(rpn_label_weight.view(1, -1))
        rpn_targets.append(rpn_target.view(1, -1, 6))
        rpn_targets_weights.append(rpn_targets_weight.view(1, -1))


    rpn_labels          = torch.cat(rpn_labels, 0)
    rpn_label_assigns   = torch.cat(rpn_label_assigns, 0)
    rpn_label_weights   = torch.cat(rpn_label_weights, 0)
    rpn_targets         = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)

    return rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights
