"""
Code adapted based on: https://github.com/HobbitLong/SupContrast.
"""
from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random

import numpy as np
import tensorboard_logger as tb_logger
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler

from utils.load_val_tiny_imagenet import Dataset
from data_aug import TwoCropTransform, aug_no_bbox_mc
from explainer import Explainer
from main_ce import set_loader as set_loader_classifier
from util import *
from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_big import SupCEResNet
from losses import SupConLoss
from main_ce import validate as validate_ce
from main_linear import validate
from infid_sen_utils import *

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns


DATA_DIR = os.getenv('SLURM_TMPDIR') if os.getenv('SLURM_TMPDIR') is not None else '../data'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1, help='seed of the run')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='the ratio of the validation set')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='the threshold for masking for drop and increase score')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['ImageNet', 'cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=DATA_DIR, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCE', 'SupCon', 'SupCon_ori', 'SimCLR', 'Ex_SupCon'], help='choose method')

    parser.add_argument('--explainer', type=str, default="GradCAM",
                        help="the explanation method to produce the augmentation")
    parser.add_argument('--explainer2', type=str, default="DeepLift")
    parser.add_argument('--negative_pair', type=int, default=0)

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--exp_epochs', type=int, default=0,
                        help='in which epoch we utilize the explainer to produce augmented images.')
    parser.add_argument('--background_anchor', type=int, default=1,
                        help='Use the background masked images as positive anchors')
    parser.add_argument('--save', type=int, default=0,
                        help='Whether to save the example images.')
    parser.add_argument('--storage_path', type=str, default="./perturbations",
                        help='A path to store the perturbed example images.')

    # Set the validation mode. Whenever in the validation mode, split a validation
    # set to do evaluation rather than using the test set.
    parser.add_argument('--validation', type=int, default=0)
    # set test mode.
    parser.add_argument('--test_mode', type=bool, default=False,
                        help='test model is True means that one is quickly running through the file to see if there is bug.')
    parser.add_argument('--fgsm', type=int, default=0,
                        help='whether to use fast gradient sign method to test adversarial robustness.')
    parser.add_argument("--eps", type=float, default=4,
                        help='scale of the adversarial ball for evaluating the adversarial robustness.')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # Recover the `negative_pair` and `background_anchor` to the default setting
    # when running the baseline.
    if "Ex" not in opt.method:
        opt.negative_pair = 0
        opt.background_anchor = 1

    if opt.negative_pair == 0:
        opt.background_anchor = 1

    print("threshold for masking for drop and increase: {}".format(opt.threshold))
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_explainer_{}_exp_epochs_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.temp, opt.trial, opt.explainer, opt.exp_epochs)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.negative_pair == 1:
        opt.model_name = '{}_negative_pair'.format(opt.model_name)
        if opt.background_anchor == 0:
            opt.model_name = '{}_background_anchor_0'.format(opt.model_name)


    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == "ImageNet":
        # The Tiny ImageNet dataset has 200 classes in all.
        opt.n_cls = 200
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    # if opt.negative_pair == 1 and 'Ex' in opt.method:
    #     opt.n_cls += 1

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == "ImageNet":
        # The mean and standard deviation statistics come from:
        # https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/ResNet18_224.ipynb
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)
    opt.normalize = normalize
    # Inverse transform reference: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    opt.inv_normalize = transforms.Compose([
        transforms.Normalize(mean = [0, 0, 0],
                             std = [1 / item for item in std]),
        transforms.Normalize(mean = [-item for item in mean],
                             std = [1, 1, 1])
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # The transformation below follows:
    #   https://github.com/pytorch/examples/blob/master/imagenet/main.py
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          train=True,
                                          download=True)
        if opt.validation == 0:
            val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                            train=False,
                                            transform=val_transform)
        elif opt.validation == 1:
            val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                            train=True,
                                            transform=val_transform)
    elif opt.dataset == "ImageNet":
        train_dataset = datasets.ImageFolder(root=opt.data_folder+"tiny-imagenet-200/train",
                                             transform=val_transform)
        # val_dataset = datasets.ImageFolder(root=opt.data_folder+"/tiny-224/val",
        #                                    transform=TwoCropTransform(standard_transform,
        #                                                               standard_transform,
        #                                                               opt))
        val_dataset = Dataset(PATH=opt.data_folder+"tiny-imagenet-200/",
                              class_to_idx=train_dataset.class_to_idx,
                              transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler, val_sampler = None, None
    if opt.validation == 1:
        # Validation set split is following a similar manner to that in:
        #   https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        train_size = len(train_dataset)
        indices = list(range(train_size))
        val_size = int(np.floor(opt.val_ratio * train_size))
        np.random.seed(opt.seed)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[val_size:], indices[:val_size]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader
    # return train_loader, val_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    encoder_state_dict = ckpt['encoder']
    classifier_state_dict = ckpt['classifier']

    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            # torch.distributed.init_process_group(backend='nccl')
            # model.encoder = torch.nn.parallel.DistributExtaParallel(model.encoder)
            model = torch.nn.DataParallel(model)
        else:
            new_encoder_state_dict = {}
            for k, v in encoder_state_dict.items():
                k = k.replace("module.", "")
                new_encoder_state_dict[k] = v
            encoder_state_dict = new_encoder_state_dict

        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # print("OKK: {}\n\n".format(encoder_state_dict))
    model.load_state_dict(encoder_state_dict)
    classifier.load_state_dict(classifier_state_dict)

    return model, classifier, criterion


def set_ce_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            # torch.distributed.init_process_group(backend='nccl')
            # model = torch.nn.parallel.DistributExtaParallel(model)
            model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)

    return model, criterion


def main():
    best_acc = 0
    opt = parse_option()

    print("===================================================================")
    print("current model: {}".format(opt.ckpt))
    print("===================================================================")

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    if "cifar" in opt.dataset:
        opt.hw = 32 * 32
    elif "ImageNet" in opt.dataset:
        opt.hw = 64 * 64
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    if "SupCon" in opt.method:
        encoder, classifier, criterion = set_model(opt)
        model = nn.Sequential(encoder.module.encoder, classifier)

    # build the explainer.
    explainer = Explainer(opt.explainer, opt.model, opt.dataset, model, opt.method)

    # 1. Evaluate the accuracy.
    loss, val_acc = validate(val_loader, encoder, classifier, criterion, opt)
    print("The top1 accuracy: {}".format(val_acc))

    # 2. Evaluate the drop score and increase score.
    eval_drop_increase(encoder, classifier, opt, val_loader, explainer)



if __name__ == '__main__':
    main()
