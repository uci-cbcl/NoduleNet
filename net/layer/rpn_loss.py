import torch
import torch.nn.functional as F
from torch import nn


def weighted_focal_loss_for_cross_entropy(logits, labels, weights, gamma=2.):
    log_probs = F.log_softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)
    probs     = F.softmax(logits, dim=1).gather(1, labels)

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum()/(weights.sum()+1e-12)

    return loss.sum()

def binary_cross_entropy_with_hard_negative_mining(logits, labels, weights, batch_size, num_hard=2):
    classify_loss = nn.BCELoss()
    probs = torch.sigmoid(logits)[:, 0].view(-1, 1)
    pos_idcs = labels[:, 0] == 1

    pos_prob = probs[pos_idcs, 0]
    pos_labels = labels[pos_idcs, 0]

    # For those weights are zero, there are 2 cases, 
    # 1. Because we first random sample num_neg negative boxes for OHEM
    # 2. Because those anchor boxes have some overlap with ground truth box, 
    #    we want to maintain high sensitivity, so we do not count those as 
    #    negative. It will not contribute to the loss
    neg_idcs = (labels[:, 0] == 0) & (weights[:, 0] != 0)
    neg_prob = probs[neg_idcs, 0]
    neg_labels = labels[neg_idcs, 0]
    if num_hard > 0:
        neg_prob, neg_labels = OHEM(neg_prob, neg_labels, num_hard * len(pos_prob))

    pos_correct = 0
    pos_total = 0
    if len(pos_prob) > 0:
        cls_loss = 0.5 * classify_loss(
            pos_prob, pos_labels.float()) + 0.5 * classify_loss(
            neg_prob, neg_labels.float())
        pos_correct = (pos_prob >= 0.5).sum()
        pos_total = len(pos_prob)
    else:
        cls_loss = 0.5 * classify_loss(
            neg_prob, neg_labels.float())


    neg_correct = (neg_prob < 0.5).sum()
    neg_total = len(neg_prob)
    return cls_loss, pos_correct, pos_total, neg_correct, neg_total


def OHEM(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


def weighted_focal_loss_with_logits(logits, labels, weights, gamma=2.):
    log_probs = F.logsigmoid(logits)
    probs     = torch.sigmoid(logits)

    pos_logprobs = log_probs[labels == 1]
    neg_logprobs = torch.log(1 - probs[labels == 0])
    pos_probs = probs[labels == 1]
    neg_probs = 1 - probs[labels == 0]
    pos_weights = weights[labels == 1]
    neg_weights = weights[labels == 0]

    pos_probs = pos_probs.detach()
    neg_probs = neg_probs.detach()

    pos_loss = - pos_logprobs * (1 - pos_probs) ** gamma
    neg_loss = - neg_logprobs * (1 - neg_probs) ** gamma
    loss = ((pos_loss * pos_weights).sum() + (neg_loss * neg_weights).sum()) / (weights.sum() + 1e-12)
    # print(pos_weights.sum())
    # print(neg_weights.sum())

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (probs[labels == 0] < 0.5).sum()
    neg_total = (labels == 0).sum()

    return loss, pos_correct, pos_total, neg_correct, neg_total

    log_probs[labels == 0] = torch.log(1 - probs[labels == 0])
    probs[labels == 0] = 1 - probs[labels == 0]

    loss = - log_probs * (1 - probs) ** gamma
    loss = (weights * loss).sum()/(weights.sum()+1e-12)

    pos_correct = (probs[labels != 0] > 0.5).sum()
    pos_total = (labels != 0).sum()
    neg_correct = (probs[labels == 0] > 0.5).sum()
    neg_total = (labels == 0).sum()

    return loss, pos_correct, pos_total, neg_correct, neg_total


def rpn_loss(logits, deltas, labels, label_weights, targets, target_weights, cfg, mode='train', delta_sigma=3.0):
    batch_size, num_windows, num_classes = logits.size()
    batch_size_k = batch_size
    labels = labels.long()

    # Calculate classification score
    pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
    batch_size = batch_size*num_windows
    logits = logits.view(batch_size, num_classes)
    labels = labels.view(batch_size, 1)
    label_weights = label_weights.view(batch_size, 1)

    # Make sure OHEM is performed only in training mode
    if mode in ['train']:
        num_hard = cfg['num_hard']
    else:
        num_hard = 10000000

    rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
        binary_cross_entropy_with_hard_negative_mining(logits, labels, \
        label_weights, batch_size_k, num_hard)
    # rpn_cls_loss, pos_correct, pos_total, neg_correct, neg_total = \
    #    weighted_focal_loss_with_logits(logits, labels, label_weights)

    # Calculate regression
    deltas = deltas.view(batch_size, 6)
    targets = targets.view(batch_size, 6)

    index = (labels != 0).nonzero()[:,0]
    deltas  = deltas[index]
    targets = targets[index]

    rpn_reg_loss = 0
    reg_losses = []
    for i in range(6):
        l = F.smooth_l1_loss(deltas[:, i], targets[:, i])
        rpn_reg_loss += l
        reg_losses.append(l.data.item())

    return rpn_cls_loss, rpn_reg_loss, [pos_correct, pos_total, neg_correct, neg_total,
                                        reg_losses[0], reg_losses[1], reg_losses[2],
                                        reg_losses[3], reg_losses[4], reg_losses[5]]
