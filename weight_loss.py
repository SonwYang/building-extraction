from functools import partial
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor


class WeightedCrossEntropy2d(nn.Module):

    def __init__(self, power=2):
        super(WeightedCrossEntropy2d, self).__init__()
        self.power = power

    def crop(self, w, h, target):
        nt, ht, wt = target.size()
        offset_w, offset_h = (wt - w) // 2, (ht - h) // 2
        if offset_w > 0 and offset_h > 0:
            target = target[:, offset_h:-offset_h, offset_w:-offset_w].clone()

        return target

    def to_one_hot(self, target, size):
        n, c, h, w = size

        ymask = torch.FloatTensor(size).zero_()
        new_target = torch.LongTensor(n, 1, h, w)
        if target.is_cuda:
            ymask = ymask.cuda(target.get_device())
            new_target = new_target.cuda(target.get_device())

        new_target[:, 0, :, :] = torch.clamp(target.detach(), 0, c - 1)
        ymask.scatter_(1, new_target, 1.0)

        return torch.autograd.Variable(ymask)

    def forward(self, input, target, weight=None):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)

        target = self.crop(w, h, target)
        ymask = self.to_one_hot(target, log_p.size())

        if weight is not None:
            weight = self.crop(w, h, weight)
            for classes in range(c):
                ymask[:, classes, :, :] = ymask[:, classes, :, :].clone() * (weight ** self.power)

        logpy = (log_p * ymask).sum(1)
        loss = -(logpy).mean()

        return loss


#########################################################################################################
class WeightedFocalLoss2d(nn.Module):
    def __init__(self, gamma=2, power=1):
        super(WeightedFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.power = power

    def crop(self, w, h, target):
        nt, ht, wt = target.size()
        offset_w, offset_h = (wt - w) // 2, (ht - h) // 2
        if offset_w > 0 and offset_h > 0:
            target = target[:, offset_h:-offset_h, offset_w:-offset_w]

        return target

    def to_one_hot(self, target, size):
        n, c, h, w = size

        ymask = torch.FloatTensor(size).zero_()
        new_target = torch.LongTensor(n, 1, h, w)
        if target.is_cuda:
            ymask = ymask.cuda(target.get_device())
            new_target = new_target.cuda(target.get_device())

        new_target[:, 0, :, :] = torch.clamp(target.detach(), 0, c - 1)
        ymask.scatter_(1, new_target, 1.0)

        return torch.autograd.Variable(ymask)

    def forward(self, input, target, weight=None):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)

        target = self.crop(w, h, target)
        ymask = self.to_one_hot(target, log_p.size())

        if weight is not None:
            weight = self.crop(w, h, weight)
            for classes in range(c):
                ymask[:, classes, :, :] = ymask[:, classes, :, :] * (weight ** self.power)

        dweight = (1 - F.softmax(input, dim=1)) ** self.gamma
        logpy = (log_p * ymask * dweight).sum(1)
        loss = -(logpy).mean()

        return loss

def multiclass_weighted_cross_entropy(output, target, weights_function=None):
    """Calculate weighted Cross Entropy loss for multiple classes.

    This function calculates torch.nn.CrossEntropyLoss(), but each pixel loss is weighted.
    Target for weights is defined as a part of target, in target[:, 1:, :, :].
    If weights_function is not None weights are calculated by applying this function on target[:, 1:, :, :].
    If weights_function is None weights are taken from target[:, 1, :, :].

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x (1 + K) x H x W). Where K is number of different weights.
        weights_function (function, optional): Function applied on target for weights.

    Returns:
        torch.Tensor: Loss value.

    """
    if weights_function is None:
        weights = target[:, 1, :, :]
    else:
        weights = weights_function(target[:, 1:, :, :])
    target = target[:, 0, :, :].long()

    loss_per_pixel = torch.nn.CrossEntropyLoss(reduce=False)(output, target)

    loss = torch.mean(loss_per_pixel * weights)
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        m2 = m2.float()
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


def dice(logits, targets):
    smooth = 1
    num = targets.size(0)
    probs = F.sigmoid(logits)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num
    return score

class WeightedBceDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedBceDiceLoss, self).__init__()

    def forward(self, logits, targets, weight=None):
        if weight is not None:
            wm = weight
        else:
            wm = torch.tensor(1.0)
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
        # targets = targets.long()
        loss_bce =  torch.nn.BCEWithLogitsLoss(reduce=False)(logits, targets)
        loss_dice = dice(logits, targets)
        loss = 1.5*torch.mean(loss_bce * wm) + 0.5 * loss_dice
        return loss


class WeightedBceLoss(nn.Module):
    def __init__(self):
        super(WeightedBceLoss, self).__init__()

    def forward(self, logits, targets, weight=None):
        if weight is not None:
            wm = weight
        else:
            wm = torch.tensor(1.0)
        logits = torch.sigmoid(logits)
        logits = torch.squeeze(logits)
        targets = torch.squeeze(targets)
        # targets = targets.long()
        loss_bce = torch.nn.BCEWithLogitsLoss(reduce=False)(logits, targets)
        return torch.mean(loss_bce * wm)
