"""
FGSM code reference: https://github.com/snu-mllab/PuzzleMix/blob/master/main.py
"""
from __future__ import print_function

import sys
import argparse
import time
import math
import numpy as np

# import tensorflow as tf
# import tensorflow_probability as tfp
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import nn
from torchvision.utils import save_image
import netcal
from netcal.metrics import ECE

from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == "ImageNet":
        # The Tiny ImageNet dataset has 200 classes in all.
        opt.n_cls = 200
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            # torch.distributed.init_process_group(backend='nccl')
            # model.encoder = torch.nn.parallel.DistributExtaParallel(model.encoder)
            model = torch.nn.DataParallel(model)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.module.encoder(images)
        output = classifier(features.detach())
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


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    final_confidence, final_labels = None, None
    # FGSM code reference: https://github.com/snu-mllab/PuzzleMix/blob/master/main.py
    print("The FGSM mode is: {}, epsilon: {}".format(opt.fgsm, opt.eps))
    for idx, (images, labels) in enumerate(val_loader):
        if opt.test_mode == True:
            if idx > 1:
                break
        images = images.float().cuda()
        if opt.dataset == "ImageNet":
            labels, bbox = labels["label"], labels["bbox"]
        labels = labels.cuda()
        bsz = labels.shape[0]

        if opt.fgsm == 1:
            # print("adding adversarial noise when calculating accuracy ...")
            input_var = Variable(images, requires_grad=True)
            # print("input variable shape: {}".format(input_var.shape))
            target_var = Variable(labels)
            # print("label shape: {}".format(target_var.shape))

            optimizer_input = torch.optim.SGD([input_var], lr=0.1)
            output = classifier(model.module.encoder(input_var))
            # print("output shape: {}".format(output.shape))
            loss = criterion(output, target_var)
            # print("loss shape: {}".format(loss.shape))
            optimizer_input.zero_grad()
            loss.backward()

            sign_data_grad = input_var.grad.sign()
            images = opt.inv_normalize(images) + opt.eps / 255. * sign_data_grad
            images = torch.clamp(images, 0, 1)
            if opt.save == 1 and idx < 20:
                for i, (image, label) in enumerate(zip(images, labels)):
                    save_image(image, opt.storage_path + "{}_{}_{}_unnormalized.png".format(idx, i, label))
                    save_image(opt.eps / 255. * sign_data_grad[i], opt.storage_path + "{}_{}_{}_noise.png".format(idx, i, label))
            images = opt.normalize(images)

        with torch.no_grad():
            input_var = Variable(images)
            target_var = Variable(labels)
            # forward
            output = classifier(model.module.encoder(input_var))
            loss = criterion(output, target_var)

            # # Calculate the expected calibration error under this batch.
            # ece = ECE()
            # calibrated_error = ece.measure(torch.softmax(output, dim=1).cpu().detach().numpy(), target_var.cpu().detach().numpy())
            # print("Expected calibration error: {}".format(calibrated_error))

            if not isinstance(final_confidence, np.ndarray):
                final_confidence = torch.softmax(output, dim=1).cpu().detach().numpy()
                final_labels = target_var.cpu().detach().numpy()
            else:
                final_confidence = np.concatenate((final_confidence, torch.softmax(output, dim=1).cpu().detach().numpy()), axis=0)
                final_labels = np.concatenate((final_labels, target_var.cpu().detach().numpy()), axis=0)
        # update metric
        losses.update(loss.item(), bsz)
        # If we are in the negative sample mode, there will be one more
        # class in the output. Thus, we need to care about only original
        # classes.
        # if opt.negative_pair == 1 and 'Ex' in opt.method:
        #     output = output[:, :opt.n_cls-1]
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

    # Calculate the overall expected calibration error.
    ece = ECE()
    calibrated_error = ece.measure(final_confidence, final_labels)
    print("Expected calibration error: {}".format(calibrated_error))

    # # Calculate the expected calibration error under TensorFlow probability.
    # logits = tf.math.log(tf.convert_to_tensor(final_confidence))
    # calibrated_error_tf = tfp.stats.expected_calibration_error(10, logits=logits, labels_true=tf.convert_to_tensor(final_labels, dtype=tf.int32))
    # print("Expected calibration error - TF: {}".format(calibrated_error_tf))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
