from . import *
import torch
import numpy as np
import torch.nn.functional as F

def mask_loss(probs, targets):
    
    loss_func = torch.nn.BCEWithLogitsLoss()
    cnt = 0
    losses = torch.zeros((len(probs))).cuda()
    weight = torch.ones((len(probs))).cuda()

    for i in range(len(probs)):
        target = targets[i]
        prob = probs[i]

        prob = prob.view(-1)
        target = target.view(-1)

        # Only those instances that their GT masks have something would contribute
        # to loss. Otherwise, too many negatives (some of them may contain false negatives).
        # We do not want to confuse the model, so just ignore those negative examples      
        if (target == 1).sum():
            prob = torch.sigmoid(prob)
            alpha = 0.5
            beta  = 0.5

            p0 = prob
            p1 = 1 - prob
            g0 = target
            g1 = 1 - target

            num = torch.sum(p0 * g0)
            den = num + alpha * torch.sum(p0 * g1) + beta * torch.sum(p1 * g0)
            
            loss = 1 - num / (den + 1e-5)
            losses[i] = loss

    return (losses * weight).sum(), losses.detach().cpu().numpy()
