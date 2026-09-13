'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import sys
import argparse
import math
import time
import shutil
import random

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.hierarchical_imagenet import (
    ImagenetHierarchihcalDataset,
    ImagenetHierarchihcalDatasetEval,
    HierarchicalBatchSampler,
)
from util.util import adjust_learning_rate, warmup_learning_rate, TwoCropTransform, WarmupCosineSchedule
from losses.losses import HMLC, HierarchicalSupervisedDCL
from network import resnet_modified
from network.resnet_modified import LinearClassifier
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_option():
    parser = argparse.ArgumentParser(description='Training/finetuning on ImageNet Dataset')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset, the superset of train/val')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'vit'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--root-dir', default='', type=str,
                        help='ImageNet root directory (containing train/ and val/ subdirs)')
    parser.add_argument('--hierarchy-file', required=True, type=str,
                        help='path to JSON mapping ImageNet class -> supercategory')
    parser.add_argument('--imagenet-index-cache', default='', type=str,
                        help='path to cache the tar-member index JSON (speeds up '
                             'subsequent runs when train/ contains .tar files)')
    parser.add_argument('--split-dir', default='', type=str,
                        help='optional subdirectory prefix under root-dir for train/val splits')
    parser.add_argument('--mode', default='train', type=str,
                        help='Train or val')
    parser.add_argument('--input-size', default=224, type=int,
                        help='input size')
    parser.add_argument('--scale-size', default=256, type=int,
                        help='scale size in validation')
    parser.add_argument('--crop-size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size per GPU (default: 512)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model (default: train from scratch for ImageNet)')
    parser.add_argument('--feature-extract', action='store_false',
                        help='When false, finetune the whole model; else only update the reshaped layer params')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,80',
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
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--loss', type=str, default='hmce',
                        help='loss type', choices=['hmc', 'hce', 'hmce'])
    parser.add_argument('--criterion', type=str, default='hmlc', choices=['hmlc', 'hsmc'],
                        help='criterion: hmlc = HMLC, hsmc = HierarchicalSupervisedDCL')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for model name')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='evaluate every N epochs (default: 1 = every epoch)')
    parser.add_argument('--amp', action='store_true',
                        help='use automatic mixed precision (fp16) to reduce GPU memory')
    args = parser.parse_args()

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    # warm-up for large-batch training
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


def concat_all_gather(tensor, rank, world_size, device):
    """All-gather a tensor across all ranks, handling variable batch sizes.

    Gradient flows through the local rank's contribution (other ranks'
    contributions are detached, as is standard for contrastive learning).

    Args:
        tensor: local tensor of shape [N_local, ...].
        rank: this process's global rank.
        world_size: total number of processes.
        device: CUDA device for this process.
    Returns:
        Concatenated tensor of shape [sum(N_i), ...] where N_i is each
        rank's local batch size.
    """
    # Step 1: exchange batch sizes (handles the +0/+1/+2 variation from
    # HierarchicalBatchSampler's triplet-based yield logic)
    local_n = torch.tensor(tensor.shape[0], device=device)
    all_ns = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(all_ns, local_n)
    all_ns_int = [int(n.item()) for n in all_ns]
    max_n = max(all_ns_int)

    # Step 2: pad to common size so all_gather works
    pad = max_n - tensor.shape[0]
    if pad > 0:
        padded = torch.cat([tensor, tensor.new_zeros(pad, *tensor.shape[1:])], dim=0)
    else:
        padded = tensor  # no copy; keeps autograd graph

    # Step 3: gather from all ranks
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # Step 4: restore gradient path for local rank's slice
    gathered[rank] = padded

    # Step 5: trim padding and concatenate
    return torch.cat([g[:n] for g, n in zip(gathered, all_ns_int)], dim=0)


def main():
    global args, best_prec1
    args = parse_option()

    # ---------------------------------------------------------------
    # Distributed setup — launched via torchrun (sets env vars
    # LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT).
    # ---------------------------------------------------------------
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    is_main = (rank == 0)
    # ---------------------------------------------------------------

    if args.seed is not None:
        # Different seed per rank → different augmentation views across GPUs
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)

    pretrained_tag = 'pretrained' if args.pretrained else 'scratch'
    args.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_{}'. \
        format(args.criterion, args.model, args.learning_rate,
               args.lr_decay_rate, args.batch_size, pretrained_tag)
    if args.tag:
        args.model_name = args.model_name + '_tag_' + args.tag

    if is_main:
        save_folder = os.path.join('./model', args.model_name)
        os.makedirs(save_folder, exist_ok=True)
    # Ensure all ranks wait until rank 0 has created the directory
    dist.barrier()

    if is_main:
        print("=> creating model '{}'".format(args.model))

    # Build model and criterion on CPU first, then move to GPU after
    # setting requires_grad flags (so DDP sees the correct param set).
    model, criterion = set_model(args)
    set_parameter_requires_grad(model, args.feature_extract)

    # SyncBatchNorm keeps BN statistics consistent across all GPUs.
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    criterion = criterion.to(device)

    # Wrap with DDP — gradient reduction happens automatically during
    # backward; find_unused_parameters=True handles frozen layers.
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = setup_optimizer(model, args.learning_rate, args.momentum,
                                args.weight_decay, args.feature_extract,
                                is_main=is_main)
    cudnn.benchmark = True

    root_dir = args.root_dir if args.root_dir else args.data
    dataloaders_dict, sampler = load_imagenet_hierarchical(
        root_dir, args.hierarchy_file, args, rank=rank, world_size=world_size)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    results = {
        'test_acc@1': [],
        'test_acc@5': [],
    }

    # Evaluate before training (epoch 0 baseline) — rank 0 only
    if is_main:
        print('Epoch 0 (before training)')
        print('-' * 10)
        test_acc_1, test_acc_5 = test(
            model.module, dataloaders_dict['memory'], dataloaders_dict['test'],
            args, epoch=0, device=device)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        pd.DataFrame(data=results, index=range(1)).to_csv(
            f'{args.model_name}_statistics.csv', index_label='epoch')

    for epoch in range(1, args.epochs + 1):
        # Advance sampler's epoch so each epoch gets a different shuffle
        sampler['train'].set_epoch(epoch)

        if is_main:
            print('Epoch {}/{}'.format(epoch, args.epochs))
            print('-' * 10)

        train(dataloaders_dict, model, criterion, optimizer, epoch, args,
              scaler, rank=rank, world_size=world_size, device=device,
              is_main=is_main)
        scheduler.step()

        # Evaluation and logging are done only on rank 0
        if is_main and (epoch <= 3 or epoch % args.eval_freq == 0 or epoch == args.epochs):
            test_acc_1, test_acc_5 = test(
                model.module, dataloaders_dict['memory'], dataloaders_dict['test'],
                args, epoch=epoch, device=device)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)

            pd.DataFrame(data=results, index=range(len(results['test_acc@1']))).to_csv(
                f'{args.model_name}_statistics.csv', index_label='epoch')

        # Keep all ranks in sync after each epoch
        dist.barrier()

    dist.destroy_process_group()


def set_model(args):
    """Build model and criterion. Does NOT move to GPU (done in main)."""
    if args.criterion == 'hsmc':
        criterion = HierarchicalSupervisedDCL(temperature=args.temp)
    else:
        criterion = HMLC(temperature=args.temp, loss_type=args.loss, layer_penalty=torch.exp)

    if args.model == 'vit':
        model = resnet_modified.MyViT(
            local_dir='pretrained_model/vit-base-patch16-224', pretrained=args.pretrained)
    else:
        model = resnet_modified.MyResNet(name='resnet50')
        if args.pretrained:
            ckpt_path = args.ckpt if args.ckpt else 'pretrained_model/resnet50-19c8e357.pth'
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model_dict = model.state_dict()
            new_state_dict = {}
            exception_list = ['fc.weight', 'fc.bias']
            for k, v in state_dict.items():
                if not k.startswith('module.head'):
                    if k in exception_list:
                        continue
                    k = 'encoder.' + k
                    new_state_dict[k] = v
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)

    return model, criterion


def train(dataloaders, model, criterion, optimizer, epoch, args, scaler=None,
          rank=0, world_size=1, device=None, is_main=True):
    """One epoch of training with cross-GPU feature aggregation before loss."""
    if device is None:
        device = torch.device('cuda')

    log_path = f'{args.model_name}_train_stats.log'
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    end = time.time()
    progress = ProgressMeter(len(dataloaders['train']),
                             [batch_time, data_time, losses],
                             prefix='Epoch: [{}]'.format(epoch))

    amp_enabled = getattr(args, 'amp', False)

    for idx, (images, labels) in enumerate(dataloaders['train']):
        data_time.update(time.time() - end)

        labels = labels.squeeze()
        images = torch.cat([images[0].squeeze(), images[1].squeeze()], dim=0)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bsz = labels.shape[0]

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            features = model(images)                                      # [2*bsz, D]
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [bsz, 2, D]

            # ----------------------------------------------------------------
            # Cross-GPU aggregation: gather features and labels from all ranks
            # before computing the loss so that each GPU's loss sees the full
            # effective batch (world_size × local_bsz samples).
            # Gradient flows back only through the local rank's slice.
            # ----------------------------------------------------------------
            if world_size > 1:
                all_features = concat_all_gather(features, rank, world_size, device)
                all_labels = concat_all_gather(labels, rank, world_size, device)
            else:
                all_features = features
                all_labels = labels

            loss = criterion(all_features, all_labels)

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        sys.stdout.flush()

        if is_main and idx % args.print_freq == 0:
            log_line = progress.display(idx)
            with open(log_path, 'a') as f:
                f.write(log_line + '\n')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_imagenet_hierarchical(root_dir, hierarchy_file, opt, rank=0, world_size=1):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.input_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    index_cache = opt.imagenet_index_cache if opt.imagenet_index_cache else None

    train_dataset = ImagenetHierarchihcalDataset(
        hierarchy_file=hierarchy_file,
        root_dir=root_dir,
        transform=TwoCropTransform(train_transform),
        index_cache=index_cache,
    )

    # Eval loaders are non-distributed: rank 0 uses the full dataset for KNN eval.
    memory_dataset = ImagenetHierarchihcalDatasetEval(
        hierarchy_file=hierarchy_file,
        root_dir=root_dir,
        transform=test_transform,
        index_cache=index_cache,
    )

    val_root = os.path.join(root_dir, 'val')
    if os.path.isdir(val_root):
        test_dataset = ImagenetHierarchihcalDatasetEval(
            hierarchy_file=hierarchy_file,
            root_dir=root_dir,
            transform=test_transform,
            index_cache=index_cache,
        )
    else:
        test_dataset = memory_dataset

    if rank == 0:
        print('LENGTH TRAIN', len(train_dataset))
        print(opt.workers, 'workers')

    # Hierarchical sampler splits the dataset across GPUs via rank/num_replicas
    train_sampler = HierarchicalBatchSampler(
        batch_size=opt.batch_size,
        drop_last=False,
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
    )
    sampler = {'train': train_sampler}

    dataloaders_dict = {}
    dataloaders_dict['train'] = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        num_workers=opt.workers, batch_size=1,
        pin_memory=True,
    )
    # Eval loaders: not distributed — rank 0 sees the full dataset
    dataloaders_dict['memory'] = torch.utils.data.DataLoader(
        memory_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.workers,
    )
    dataloaders_dict['test'] = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.workers,
    )

    return dataloaders_dict, sampler


def setup_optimizer(model_ft, lr, momentum, weight_decay, feature_extract, is_main=True):
    if feature_extract:
        params_to_update = [p for p in model_ft.parameters() if p.requires_grad]
        if is_main:
            print('Params to learn:')
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    print('\t', name)
    else:
        params_to_update = list(model_ft.parameters())
        if is_main:
            print('Params to learn: all')

    return torch.optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)


def set_parameter_requires_grad(model, feature_extracting):
    if hasattr(model, 'module'):
        model = model.module
    if feature_extracting:
        is_vit = isinstance(model, resnet_modified.MyViT)
        for name, param in model.named_parameters():
            if is_vit:
                if any(name.startswith(f'encoder.vit.encoder.layer.{i}') for i in [9, 10, 11]):
                    param.requires_grad = True
                elif name.startswith('encoder.vit.layernorm'):
                    param.requires_grad = True
                elif name.startswith('head'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
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
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_line = '\t'.join(entries)
        print(log_line)
        return log_line

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
            label_bank.append(labels[1])  # use fine (ImageNet class) label
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(label_bank, dim=0).to(device)

        # Normalize the feature bank once before the test loop
        feature_bank = torch.nn.functional.normalize(feature_bank, dim=0)  # [D, N]

        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data = data.to(device, non_blocking=True)
            target = target[1].to(device, non_blocking=True)
            feature = net.encoder(data)

            total_num += data.size(0)

            # L2-normalize query features
            feature = torch.nn.functional.normalize(feature, dim=1)  # [B, D]

            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=args.k, dim=-1)
            sim_weight = (sim_weight / args.temp).exp()
            # [B, K]
            sim_labels = torch.gather(
                feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)

            # counts for each class
            one_hot_label = torch.zeros(
                data.size(0) * args.k, args.num_classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(data.size(0), -1, args.num_classes) * sim_weight.unsqueeze(dim=-1),
                dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.5f}% Acc@5:{:.5f}%'.format(
                    epoch, args.epochs,
                    total_top1 / total_num * 100,
                    total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    main()