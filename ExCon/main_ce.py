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
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler

from explainer import Explainer
from utils.load_val_tiny_imagenet import Dataset
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model_ce
from util import eval_drop_increase
from networks.resnet_big import SupCEResNet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

# DATA_DIR = os.getenv('SLURM_TMPDIR') if os.getenv('SLURM_TMPDIR') is not None else '../data'
DATA_DIR = None

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1, help='seed of the run')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='the ratio of the validation set')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
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
                        choices=['SVHN', 'cifar10', 'cifar100', 'ImageNet'], help='dataset')

    # other setting
    parser.add_argument('--explainer', type=str, default="GradCAM",
                        help="the explanation method to produce the augmentation")
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # Set the validation mode. Whenever in the validation mode, split a validation
    # set to do evaluation rather than using the test set.
    parser.add_argument('--validation', type=int, default=0)
    # set test mode.
    parser.add_argument('--test_mode', type=bool, default=False,
                        help='test model is True means that one is quickly running through the file to see if there is bug.')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_explainer_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial, opt.explainer)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.validation == 1:
        opt.model_name = '{}_validation'.format(opt.model_name)

    opt.model_name = '{}_seed_{}'.format(opt.model_name, opt.seed)

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

    if opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == "ImageNet":
        # The Tiny ImageNet dataset has 200 classes in all.
        opt.n_cls = 200
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar100':
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


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            # torch.distributed.init_process_group(backend='nccl')
            # model = torch.nn.parallel.DistributedDataParallel(model)
            model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            if opt.test_mode == True:
                if idx > 1:
                    break
            images = images.float().cuda()
            if opt.dataset == "ImageNet":
                labels, bbox = labels["label"], labels["bbox"]
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print("===================================================================")
    print("current hyperparameters: {}".format(opt.model_name))
    print("===================================================================")
    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build the explainer.
    explainer = Explainer(opt.explainer, opt.model, opt.dataset, model, "SupCE")

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        if opt.test_mode == True:
            if epoch > 1:
                break
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        print("\nvalidation accuracy: {}\n".format(val_acc))
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            old_best_acc = best_acc
            best_acc = val_acc
            del_file = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(old_best_acc))
            if os.path.isfile(del_file):
                print("delete {}".format(del_file))
                os.remove(del_file)
            save_file = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(best_acc))
            print("save {}".format(save_file))
            save_model_ce(model, optimizer, opt, opt.epochs, save_file)


    eval_drop_increase(model, None, opt, val_loader, explainer)
    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model_ce(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
