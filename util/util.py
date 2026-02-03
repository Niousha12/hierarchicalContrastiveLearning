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


class WarmupCosineSchedule(object):
    def __init__(
            self, optimizer, warmup_steps, lr_start, lr_ref, T_max, last_epoch=-1, lr_final=0.0, warmup_mode="linear"
    ):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_ref = lr_ref
        self.lr_final = lr_final
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            if self.warmup_mode == "linear":
                new_lr = self.lr_start + progress * (self.lr_ref - self.lr_start)
            elif self.warmup_mode == "cosine":
                new_lr = self.lr_start + (self.lr_ref - self.lr_start) * 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                raise ValueError(f"Unrecognized warmup shape: {self.warmup_mode}")
        else:
            # -- progress after warmup
            progress = min(1.0, float(self._step - self.warmup_steps) / float(max(1, self.T_max - self.warmup_steps)))
            new_lr = max(self.lr_final,
                         self.lr_final + (self.lr_ref - self.lr_final) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                         )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr * group.get("lr_scale", 1.0)

        return new_lr


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
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
