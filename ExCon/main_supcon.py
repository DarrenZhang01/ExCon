"""
Code adapted based on: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py.
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
from util import adjust_learning_rate, warmup_learning_rate, AverageMeter
from util import set_optimizer, save_model, accuracy
from util import eval_drop_increase
from networks.resnet_big import SupConResNet, LinearClassifier
from main_linear import validate
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

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
                        choices=['ImageNet', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=DATA_DIR, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SupCon_ori', 'SimCLR', 'Ex_SupCon', 'Ex_SimCLR'], help='choose method')

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

    opt = parser.parse_args()

    # Recover the `negative_pair` and `background_anchor` to the default setting
    # when running the baseline.
    if "Ex" not in opt.method:
        opt.negative_pair = 0
        opt.background_anchor = 1

    if opt.negative_pair == 0:
        opt.background_anchor = 1

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

    # if opt.negative_pair == 1 and 'Ex' in opt.method:
    #     opt.n_cls += 1

    return opt


def set_loader_encoder(opt):
    print("validation: {}".format(opt.validation))
    # construct data loader
    if opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == "ImageNet":
        # The mean and standard deviation statistics come from:
        # https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/ResNet18_224.ipynb
        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2302, 0.2265, 0.2262)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
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
    random_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    standard_transform = transforms.Compose([
        transforms.Resize(opt.size),
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

    val_dataset = None

    if opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(random_transform,
                                                                     standard_transform,
                                                                     opt),
                                          train=True,
                                          download=True)
        if opt.validation == 1:
            val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                            transform=TwoCropTransform(standard_transform,
                                                                       standard_transform,
                                                                       opt),
                                            train=True,
                                            download=True)
    elif opt.dataset == 'path' or opt.dataset == "ImageNet":
        train_dataset = datasets.ImageFolder(root=opt.data_folder+"tiny-imagenet-200/train",
                                             transform=TwoCropTransform(random_transform,
                                                                        standard_transform,
                                                                        opt))
        # val_dataset = datasets.ImageFolder(root=opt.data_folder+"/tiny-224/val",
        #                                    transform=TwoCropTransform(standard_transform,
        #                                                               standard_transform,
        #                                                               opt))
        val_dataset = Dataset(PATH=opt.data_folder+"tiny-imagenet-200/",
                              class_to_idx=train_dataset.class_to_idx,
                              transform=TwoCropTransform(standard_transform,
                                                         standard_transform,
                                                         opt))
    else:
        raise ValueError(opt.dataset)

    train_sampler, val_loader = None, None
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

    if opt.validation == 1:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            # torch.distributed.init_process_group(backend='nccl')
            # model.encoder = torch.nn.parallel.DistributExtaParallel(model.encoder)
            model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_classifier(opt):
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    if torch.cuda.is_available():
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return classifier, criterion


def train_encoder(train_loader,
                  encoder,
                  classifier,
                  criterion,
                  optimizer,
                  epoch,
                  opt,
                  explainer1):
    """one epoch training

    The classifier here is used to produce explanations.
    """
    encoder.train()
    classifier.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        if opt.test_mode == True:
            if idx > 1:
                break
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images[0] = images[0].cuda(non_blocking=True)
            images[1] = images[1].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        # Record the batch size before the data augmentation, i.e., before
        # potentially append the background masked images.
        actual_bsz = images[0].shape[0]
        # print("size1: {}, size2:{} , size3: {}".format(images[0].shape, images[1].shape, images[2].shape))

        ### -------- Augment the frst view of the images based on Ex --------
        if 'Ex' in opt.method and epoch < opt.exp_epochs:
            images[2] = images[2].cuda(non_blocking=True)
            images[0] = images[2]
        elif 'Ex' in opt.method and epoch >= opt.exp_epochs:
            # The back-up randomly cropped image when the masked image gives the wrong prediction.
            images[2] = images[2].cuda(non_blocking=True)
            # images[3] = images[3].cuda(non_blocking=True)
            model = lambda x: classifier(encoder.module.encoder(x))
            # cache1 = torch.clone(images[0])
            # cache2 = torch.clone(images[1])
            tuple = aug_no_bbox_mc(images[0],
                                   images[1],
                                   torch.clone(labels),
                                   explainer1,
                                   model,
                                   -1, 0, 0.5,
                                   images[2],
                                   opt)
            images[0], images[1], labels, flag1 = tuple
        ### -------------------------------------------------------------------

        images = torch.cat([images[0], images[1]], dim=0)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = encoder(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # print("features: {}, labels: {}".format(features.shape, labels.shape))
        # print("labels: {}".format(labels))
        if 'SupCon' in opt.method:
            loss = criterion(features, opt, labels, actual_bsz=actual_bsz)
        elif 'SimCLR' in opt.method:
            loss = criterion(features, opt, actual_bsz=actual_bsz)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("encoder: {}".format(encoder))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def train_classifier(train_loader, encoder, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    encoder.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        if opt.test_mode == True:
            if idx > 1:
                break
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = encoder.module.encoder(images)
        output = classifier(features.detach())
        # print("output shape: {}, labels shape: {}".format(output.shape, labels.shape))
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


# TODO: Embed the training of the linear classifier under each epoch.
def main():
    best_acc = 0
    opt = parse_option()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print("===================================================================")
    print("current hyperparameters: {}".format(opt.model_name))
    print("===================================================================")
    # build data loaders.
    train_loader_encoder, _ = set_loader_encoder(opt)
    train_loader_classifier, val_loader_classifier = set_loader_classifier(opt)

    # build the encoder and the according criterion.
    encoder, e_criterion = set_model(opt)

    # build the classifier and the according criterion.
    classifier, c_criterion = set_classifier(opt)

    # build optimizers.
    e_optimizer = set_optimizer(opt, encoder)
    c_optimizer = set_optimizer(opt, classifier)

    # build the explainer.
    model = nn.Sequential(encoder.module.encoder, classifier)
    explainer1 = Explainer(opt.explainer, opt.model, opt.dataset, model, opt.method)
    # explainer2 = Explainer(opt.explainer2, opt.model, opt.dataset, model, opt.method)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # Use a list to save the validation accuracies along the training epochs.
    accuracies = []
    # training routine
    if opt.method == "SupCon_ori":
        for epoch in range(1, opt.epochs + 1):
            # -------------------- Train the encoder. -----------------------
            adjust_learning_rate(opt, e_optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            # loss = train_encoder(train_loader_encoder, encoder, classifier, e_criterion, e_optimizer, epoch, opt, explainer1, explainer2)
            loss = train_encoder(train_loader_encoder, encoder, classifier, e_criterion, e_optimizer, epoch, opt,
                                 explainer1)
            time2 = time.time()
            print('epoch {}, total time (encoder) {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss', loss, epoch)
            logger.log_value('learning_rate', e_optimizer.param_groups[0]['lr'], epoch)

            """
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(encoder, e_optimizer, classifier, c_optimizer, opt, epoch, save_file)
            """

        # ---------------------- Train the classifier. -----------------------
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, c_optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss = train_classifier(train_loader_classifier, encoder, classifier, c_criterion, c_optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time (classifier) {:.2f}'.format(epoch, time2 - time1))

            # eval for one epoch
            loss, val_acc = validate(val_loader_classifier, encoder, classifier, c_criterion, opt)
            logger.log_value("val_acc", val_acc, epoch)
            print("\nvalidation accuracy: {}\n".format(val_acc))

            if val_acc > best_acc:
                old_best_acc = best_acc
                best_acc = val_acc
                del_file = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(old_best_acc))
                if os.path.isfile(del_file):
                    print("delete {}".format(del_file))
                    os.remove(del_file)
                save_file = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(best_acc))
                print("save {}".format(save_file))
                save_model(encoder, e_optimizer, classifier, c_optimizer, opt, epoch, save_file)

            accuracies.append(val_acc)
    else:
        for epoch in range(1, opt.epochs + 1):
            # -------------------- Train the encoder. -----------------------
            adjust_learning_rate(opt, e_optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            # loss = train_encoder(train_loader_encoder, encoder, classifier, e_criterion, e_optimizer, epoch, opt, explainer1, explainer2)
            loss = train_encoder(train_loader_encoder, encoder, classifier, e_criterion, e_optimizer, epoch, opt, explainer1)
            time2 = time.time()
            print('epoch {}, total time (encoder) {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('loss', loss, epoch)
            logger.log_value('learning_rate', e_optimizer.param_groups[0]['lr'], epoch)

            # ---------------------- Train the classifier. -----------------------
            adjust_learning_rate(opt, c_optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss = train_classifier(train_loader_classifier, encoder, classifier, c_criterion, c_optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time (classifier) {:.2f}'.format(epoch, time2 - time1))

            # eval for one epoch
            loss, val_acc = validate(val_loader_classifier, encoder, classifier, c_criterion, opt)
            logger.log_value("val_acc", val_acc, epoch)
            print("\nvalidation accuracy: {}\n".format(val_acc))

            if val_acc > best_acc:
                old_best_acc = best_acc
                best_acc = val_acc
                del_file = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(old_best_acc))
                if os.path.isfile(del_file):
                    print("delete {}".format(del_file))
                    os.remove(del_file)
                save_file = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(best_acc))
                print("save {}".format(save_file))
                save_model(encoder, e_optimizer, classifier, c_optimizer, opt, epoch, save_file)

            accuracies.append(val_acc)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(encoder, e_optimizer, classifier, c_optimizer, opt, opt.epochs, save_file)

    ### load best model###
    best_model_path = os.path.join(opt.save_folder, 'ckpt_best_val_acc_{}.pth'.format(best_acc))
    print("load best model_{}".format(best_model_path))
    best_model = torch.load(best_model_path)
    encoder.load_state_dict(best_model['encoder'])
    classifier.load_state_dict(best_model['classifier'])
    opt = best_model['opt']

    eval_drop_increase(encoder, classifier, opt, val_loader_classifier, explainer1)

    if 'ImageNet' in opt.dataset:
        eval_ebpg_miou_bbox(val_loader_classifier, explainer1)

    print('best accuracy: {:.2f}'.format(best_acc))

    accuracies = [accuracy.item() for accuracy in accuracies]
    accuracies = np.asarray(accuracies)
    file_name = "./" + opt.dataset + "_" + opt.method + "_batch_size_" + str(opt.batch_size) + ".npy"
    np.save(file_name, accuracies)


if __name__ == '__main__':
    main()
