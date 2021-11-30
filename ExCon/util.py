"""
References:
    https://github.com/HobbitLong/SupContrast/blob/master/util.py
"""


from __future__ import print_function

import math
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from itertools import permutations
from scipy.ndimage.filters import gaussian_filter
import torchvision
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(encoder, e_optimizer, classifier, c_optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'encoder': encoder.state_dict(),
        'e_optimizer': e_optimizer.state_dict(),
        'classifier': classifier.state_dict(),
        'c_criterion': c_optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def save_model_ce(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state



def target_prob(encoder, classifier, input, target, dataset):
    ori = None
    if classifier is None:
        ori = torch.softmax(encoder(input.unsqueeze(0))[0], dim=0)[target].item()
    else:
        ori = torch.softmax(classifier(encoder.module.encoder(input.unsqueeze(0)))[0], dim=0)[target].item()
    return ori


def umask_embeddings(encoder, classifier, inputs, targets, explainer):
    """
    Produce unimportant masks according to the importance scores of the input pixels.
    Then get their encoder embeddings.
    """
    embeddings = []
    for i, (_input, _target) in enumerate(zip(inputs, targets)):
        max_label = None
        # If `classifier` is None, we are in the state of cross entropy training.
        if classifier is None:
            max_label = torch.argmax(encoder(_input.unsqueeze(0))[0])
        else:
            max_label = torch.argmax(classifier(encoder.module.encoder(_input.unsqueeze(0)))[0])
        attributions = explainer.attribute(torch.unsqueeze(_input, 0), max_label)

        exp_flat = attributions.flatten()
        sorted_idx = np.argsort(exp_flat * -1)[:int(0.6*len(attributions.flatten()))]
        exp = np.zeros_like(attributions).flatten()
        exp[sorted_idx] = 1
        exp = exp.reshape(attributions.shape[0], attributions.shape[1])

        embedding = encoder.module.encoder((_input * torch.from_numpy(np.tile(exp, (3, 1, 1))).cuda()).unsqueeze(0))
        embeddings.append(embedding)

    # Perturb the targets based on the original targets such that we can
    # distinguish the masked images from the original images.
    return torch.cat(embeddings, dim=0), targets * (-1) - 1


def drop_increase_rate(threshold, encoder, classifier, input, target, cam, dataset, degree_change=True, increase=True):
    ori = target_prob(encoder, classifier, input, target, dataset)
    exp_flat = cam.flatten()
    # sorted_idx = np.argsort(exp_flat * -1)[:int(0.15*len(cam.flatten()))]
    # sorted_idx = np.argsort(exp_flat * -1)[:int(0.3*len(cam.flatten()))]
    sorted_idx = np.argsort(exp_flat * -1)[:int(threshold * len(cam.flatten()))]
    exp = np.zeros_like(cam).flatten()
    exp[sorted_idx] = 1
    exp = exp.reshape(cam.shape[0], cam.shape[1])
    if classifier is None:
        pert = torch.softmax(encoder((input * torch.from_numpy(np.tile(exp, (3, 1, 1))).cuda()).unsqueeze(0))[0], dim=0)[target].item()
    else:
        pert = torch.softmax(classifier(encoder.module.encoder((input * torch.from_numpy(np.tile(exp, (3, 1, 1))).cuda()).unsqueeze(0)))[0], dim=0)[target].item()
    if increase is True:
        if degree_change is True:
            rate = np.maximum(0, pert - ori) / ori
        elif degree_change is False:
            rate = int(pert > ori)
    elif increase is False:
        if degree_change is True:
            rate = np.maximum(0, ori - pert) / ori
        elif degree_change is False:
            rate = int(ori > pert)
    return rate


def eval_drop_increase(encoder, classifier, opt, testloader, explainer):
    encoder.eval()
    if classifier is not None:
        classifier.eval()
    # The order of the three elements will be "degree_change" and "increase".
    perm_list = list(set(permutations([True, False, True, False], 2)))
    rate_dict = {}
    for combo in perm_list:
        rate_dict[" ".join([str(item) for item in combo])] = []
    for batch_idx, batch_data in enumerate(testloader):
        if opt.dataset == 'voc':
            inputs, targets, bbox = batch_data
        else:
            inputs, targets = batch_data
        if opt.dataset == "ImageNet":
            targets, bbox = targets["label"], targets["bbox"]
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        for i, (_input, _target) in enumerate(zip(inputs, targets)):
            max_label = None
            # If `classifier` is None, we are in the state of cross entropy training.
            if classifier is None:
                max_label = torch.argmax(encoder(_input.unsqueeze(0))[0])
            else:
                max_label = torch.argmax(classifier(encoder.module.encoder(_input.unsqueeze(0)))[0])
            attributions = explainer.attribute(torch.unsqueeze(_input, 0), max_label)
            for combo in perm_list:
                rate = drop_increase_rate(opt.threshold, encoder, classifier, _input, max_label,
                                          attributions, opt.dataset, combo[0], combo[1])
                rate_dict[" ".join([str(item) for item in combo])].append(rate)
    for combo in perm_list:
        items = rate_dict[" ".join([str(item) for item in combo])]
        avg = sum(items) / len(items)
        print("degree_change {} increase {}: {}".format(
            combo[0], combo[1], avg
        ))
