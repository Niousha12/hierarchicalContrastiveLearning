'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import sys
import argparse

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.generate_dataset import DatasetCategory
from data_processing.hierarchical_dataset import DeepFashionHierarchihcalDataset, HierarchicalBatchSampler, \
    DeepFashionHierarchihcalDatasetEval
from torch.optim import lr_scheduler
from util.util import adjust_learning_rate, warmup_learning_rate, TwoCropTransform, WarmupCosineSchedule
from losses.losses import HMLC, HierarchicalSupervisedDCL
from network import resnet_modified
from network.resnet_modified import LinearClassifier
# import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import shutil
import math
import builtins


def parse_option():
    parser = argparse.ArgumentParser(description='Training/finetuning on Deep Fashion Dataset')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train-listfile', default='', type=str,
                        help='training file with annotation')
    parser.add_argument('--val-listfile', default='', type=str,
                        help='validation file with annotation')
    parser.add_argument('--test-listfile', default='', type=str,
                        help='test file with annotation')
    parser.add_argument('--class-map-file', default='', type=str,
                        help='class mapping between str and int')
    parser.add_argument('--class-seen-file', default='', type=str,
                        help='seen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--class-unseen-file', default='', type=str,
                        help='unseen classes text file. Used for seen/unseen split experiments.')
    parser.add_argument('--repeating-product-file', default='', type=str,
                        help='repeating product ids file')
    parser.add_argument('--mode', default='train', type=str,
                        help='Train or val')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--num-classes', type=int,
                        help='number of classes')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 512)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                        help='use pre-trained model')
    parser.add_argument('--feature-extract', action='store_false',
                        help='When flase, finetune the whole model; else only update the reshaped layer para')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--loss', type=str, default='hmce',
                        help='loss type', choices=['hmc', 'hce', 'hmce'])
    parser.add_argument('--tag', type=str, default='',
                        help='tag for model name')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    # warm-up for large-batch training,
    if args.batch_size >= 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    return args


best_prec1 = 0


def main():
    global args, best_prec1
    args = parse_option()

    args.save_folder = './model'
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    args.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_loss_{}_trial_{}'. \
        format('hmlc', 'dataset', args.model, args.learning_rate,
               args.lr_decay_rate, args.batch_size, args.loss, 5)
    if args.tag:
        args.model_name = args.model_name + '_tag_' + args.tag
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=> creating model '{}'".format(args.model))
    model, criterion = set_model(device, args)

    set_parameter_requires_grad(model, args.feature_extract)
    optimizer = setup_optimizer(model, args.learning_rate, args.momentum, args.weight_decay, args.feature_extract)
    cudnn.benchmark = True

    dataloaders_dict, sampler = load_deep_fashion_hierarchical(args.data, args.train_listfile,
                                                               args.val_listfile, args.test_listfile,
                                                               args.class_map_file, args.repeating_product_file, args)

    ##########
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-4)
    total_steps = args.epochs * len(dataloaders_dict['train'])
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=int(0.1 * total_steps),  # 5% warmup is common
                                     lr_start=0.0, lr_ref=args.learning_rate, T_max=total_steps, lr_final=0.0,
                                     warmup_mode="linear", )
    ##########

    results = {
        'test_acc@1': [],
        'test_acc@5': [],
    }
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs + 1))
        print('-' * 10)
        # adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(dataloaders_dict, model, criterion, optimizer, scheduler, epoch, args)
        # scheduler.step()

        test_acc_1, test_acc_5 = test(model, dataloaders_dict['memory'], dataloaders_dict['test'], args, epoch=epoch,
                                      device='cuda')
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'statistics_1.csv', index_label='epoch')

        # To save checkpoint, uncomment the following lines
        # output_file = args.save_folder + '/checkpoint_{:04d}.pth.tar'.format(epoch)
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.model,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, is_best=False,
        #     filename=output_file)


def set_model(device, args):
    model = resnet_modified.MyResNet(name='resnet50')
    # criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)
    criterion = HierarchicalSupervisedDCL(temperature=args.temp)

    # This part is to load a pretrained model
    state_dict = torch.load("../pretrained_model/resnet50-19c8e357.pth", map_location='cpu', weights_only=False)
    model_dict = model.state_dict()
    new_state_dict = {}
    exception_list = ["fc.weight", "fc.bias"]
    for k, v in state_dict.items():
        if not k.startswith('module.head'):
            # k = k.replace('module.encoder', 'encoder')
            if k in exception_list:
                continue
            k = 'encoder.' + k
            new_state_dict[k] = v
    state_dict = new_state_dict
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    model = model.to(device)
    criterion = criterion.to(device)

    return model, criterion


def train(dataloaders, model, criterion, optimizer, scheduler, epoch, args):
    """
    one epoch training
    """
    log_path = "train_stats.log"
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()

    progress = ProgressMeter(len(dataloaders['train']),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()  # Set model to training mode

    # Iterate over data.
    for idx, (images, labels) in enumerate(dataloaders['train']):
        data_time.update(time.time() - end)
        labels = labels.squeeze()
        images = torch.cat([images[0].squeeze(), images[1].squeeze()], dim=0)
        images = images.cuda(non_blocking=True)
        labels = labels.squeeze().cuda(non_blocking=True)
        bsz = labels.shape[0]  # batch size
        # warmup_learning_rate(args, epoch, idx, len(dataloaders['train']), optimizer)

        # forward
        # track history if only in train
        # Get model outputs and calculate loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        sys.stdout.flush()
        if idx % args.print_freq == 0:
            log_line = progress.display(idx)

            with open(log_path, "a") as f:
                f.write(log_line + "\n")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_data(root_dir, train_listfile, val_listfile, class_map_file,
              class_seen_file, class_unseen_file, input_size, scale_size, crop_size, batch_size, distributed, workers):
    # Data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # Create training and validation datasets
    train_dataset = DatasetCategory(root_dir, 'train', train_listfile, val_listfile, '',
                                    class_map_file, class_seen_file,
                                    class_unseen_file, TwoCropTransform(data_transforms['train']))
    val_dataset = DatasetCategory(root_dir, 'val', train_listfile, val_listfile, '',
                                  class_map_file, class_seen_file,
                                  class_unseen_file, TwoCropTransform(data_transforms['val']))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset}
    print("Initializing Datasets and Dataloaders...")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    sampler = {'train': train_sampler,
               'val': val_sampler}

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=(train_sampler is None), num_workers=workers,
                                       pin_memory=True, sampler=sampler[x], drop_last=True)
        for x in ['train', 'val']}
    return dataloaders_dict, sampler


def load_deep_fashion_hierarchical(root_dir, train_list_file, val_list_file, test_list_file, class_map_file,
                                   repeating_product_file, opt):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.input_size, scale=(0.8, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4)
        ], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [
                                            0.229, 0.224, 0.225]),
                                        ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = DeepFashionHierarchihcalDataset(os.path.join(root_dir, train_list_file),
                                                    os.path.join(root_dir, class_map_file),
                                                    os.path.join(root_dir, repeating_product_file),
                                                    transform=TwoCropTransform(train_transform))

    val_dataset = DeepFashionHierarchihcalDataset(os.path.join(root_dir, val_list_file),
                                                  os.path.join(root_dir, class_map_file),
                                                  os.path.join(root_dir, repeating_product_file),
                                                  transform=TwoCropTransform(val_transform))

    test_dataset = DeepFashionHierarchihcalDatasetEval(os.path.join(root_dir, test_list_file),
                                                       os.path.join(root_dir, class_map_file),
                                                       os.path.join(root_dir, repeating_product_file),
                                                       transform=test_transform)

    memory_dataset = DeepFashionHierarchihcalDatasetEval(os.path.join(root_dir, train_list_file),
                                                         os.path.join(root_dir, class_map_file),
                                                         os.path.join(root_dir, repeating_product_file),
                                                         transform=test_transform)

    print('LENGTH TRAIN', len(train_dataset))
    image_datasets = {'train': train_dataset,
                      'val': val_dataset,
                      'test': test_dataset,
                      'memory': memory_dataset}
    train_sampler = HierarchicalBatchSampler(batch_size=opt.batch_size,
                                             drop_last=False,
                                             dataset=train_dataset)
    val_sampler = HierarchicalBatchSampler(batch_size=opt.batch_size,
                                           drop_last=False,
                                           dataset=val_dataset)
    sampler = {'train': train_sampler,
               'val': val_sampler}
    print(opt.workers, "workers")
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], sampler=sampler[x],
                                       num_workers=opt.workers, batch_size=1,
                                       pin_memory=True)
        for x in ['train', 'val']}

    dataloaders_dict['memory'] = torch.utils.data.DataLoader(image_datasets['memory'], batch_size=opt.batch_size,
                                                             shuffle=False, num_workers=16)
    dataloaders_dict['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=opt.batch_size,
                                                           shuffle=False, num_workers=16)

    return dataloaders_dict, sampler


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract):
    # Send the model to GPU
    # model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    # optimizer_ft = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer_ft = torch.optim.AdamW(params_to_update, lr=lr, weight_decay=weight_decay)
    return optimizer_ft


def set_parameter_requires_grad(model, feature_extracting):
    if hasattr(model, "module"):
        model = model.module
    if feature_extracting:
        for name, param in model.named_parameters():
            if name.startswith('encoder.layer4'):
                param.requires_grad = True
            elif name.startswith('encoder.layer3'):
                param.requires_grad = True
            elif name.startswith('head'):
                param.requires_grad = True
            else:
                param.requires_grad = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_line = '\t'.join(entries)  # exact console format

        print(log_line)  # still print to terminal
        return log_line  # return to caller for logging

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def test(net, memory_data_loader, test_data_loader, args, epoch, device):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, label_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, labels in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net.encoder(data.to(device, non_blocking=True))
            feature_bank.append(feature)
            # labels = torch.zeros_like(labels[0])
            label_bank.append(labels[0])  # TODO: category is 0
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(label_bank, dim=0)
        feature_labels = feature_labels.to(device)
        # [N]
        # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            # TODO: category is 0
            data, target = data.to(device, non_blocking=True), target[0].to(device, non_blocking=True)
            feature = net.encoder(data)

            total_num += data.size(0)

            # L2-normalize along feature dimension
            feature = torch.nn.functional.normalize(feature, dim=1)  # [B, D]
            feature_bank = torch.nn.functional.normalize(feature_bank, dim=0)  # [D, K]

            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            sim_weight = (sim_weight / args.temp).exp()
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.k, args.num_classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, args.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.5f}% Acc@5:{:.5f}%'
                                     .format(epoch, args.epochs, total_top1 / total_num * 100,
                                             total_top5 / total_num * 100))

            # break
    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    main()
