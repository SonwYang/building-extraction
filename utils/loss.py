"""PyTorch-compatible losses and loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



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


class CrossEntropyLoss2d(nn.Module):
    """Cross-entropy.

    See: http://cs231n.github.io/neural-networks-2/#losses
    """

    def __init__(self, weight=None):
        """Creates an `CrossEntropyLoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(nn.Module):
    """Focal Loss.

    Reduces loss for well-classified samples putting focus on hard mis-classified samples.

    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.

        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight,reduce=False)
        self.gamma = gamma

    def forward(self, inputs, targets):
        penalty = (1 - nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(penalty * nn.functional.log_softmax(inputs, dim=1), targets)

class ComboLoss(nn.Module):
    """Focal Loss.

    Reduces loss for well-classified samples putting focus on hard mis-classified samples.

    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.

        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)
        self.gamma = gamma

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

    def forward(self, outs, fc=None, labels=None):
        size = outs.size()
        masks = self.to_one_hot(labels, size)
        if fc is None:
            loss = 256 * nn.BCEWithLogitsLoss(reduce=True)(outs, masks)
            # loss = lovasz_hinge(outs, labels)
        else:
            b = len(outs)
            loss = 256 * nn.BCEWithLogitsLoss(reduce=True)(outs, masks)
            labels_fc = (labels.view(b, -1).sum(-1) > 0).float().view(b, 1)
            loss += nn.BCEWithLogitsLoss(reduce=True)(fc, labels_fc)
            # loss += 32 * FocalLoss(2.0)(fc, labels_fc)
        return loss

class mIoULoss2d(nn.Module):
    """mIoU Loss.

    See:
      - http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      - http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf
    """

    def __init__(self, weight=None):
        """Creates a `mIoULoss2d` instance.

        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()

        softs = nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1).permute(1, 0, 2, 3)
        # print("mIoULoss2d-------outputs'size:{},masks'size:{}".format(softs.size(),masks.size()))

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)

        miou = 1. - (inters.view(C, N, -1).sum(2) / unions.view(C, N, -1).sum(2)).mean()

        return max(miou, self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets))


class LovaszLoss2d(nn.Module):
    """Lovasz Loss.

    See: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        """Creates a `LovaszLoss2d` instance."""
        super().__init__()

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1)

        loss = 0.

        for mask, input in zip(masks.view(N, -1), inputs.view(N, -1)):

            max_margin_errors = 1. - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1. - labels_sorted).cumsum(0)
            iou = 1. - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]

            loss += torch.dot(nn.functional.relu(errors_sorted), iou)

        return loss / N
