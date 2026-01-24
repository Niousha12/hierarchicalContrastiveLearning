from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    T1 = 100
    T2 = args.epochs - T1
    if args.cosine:
        eta_mid = lr * (args.lr_decay_rate ** 3)
        eta_min = lr * (args.lr_decay_rate ** 4)

        # lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2

        if epoch <= T1:
            lr = eta_mid + (lr - eta_mid) * (1 + math.cos(math.pi * epoch / T1)) / 2
        else:
            e2 = epoch - T1
            lr = eta_min + (eta_mid - eta_min) * (1 + math.cos(math.pi * e2 / T2)) / 2

    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer
