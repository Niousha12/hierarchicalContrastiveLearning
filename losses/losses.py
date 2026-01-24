'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from __future__ import print_function

import torch
import torch.nn as nn


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        for l in range(1, labels.shape[1]):  # start from fine label
            mask[:, labels.shape[1] - l:] = 0  # mask finer labels
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)  # matrix
            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1 / (l)).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1 / l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        return cumulative_loss / labels.shape[1]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class HierarchicalSupervisedDCL(nn.Module):
    def __init__(self, temperature=0.07, clamp=True, augmentation=False, agg_type="mean"):
        super(HierarchicalSupervisedDCL, self).__init__()
        self.temperature = temperature
        self.clamp_negative = clamp
        self.augmentation = augmentation
        self.agg_type = agg_type
        self.is_symmetric = False

    def supervised_contrastive_learning(self, pos_sim, neg_sim, labels, level, device, dtype):
        """
        Compute supervised contrastive learning loss (equal to eq. 10):
        A cluster-wise logsumexp loss that encourages intra-cluster similarity (positive pairs)
        while penalizing inter-cluster similarity (negative pairs). Clusters with fewer than two members are skipped.

        Input:
            pos_sim: Pairwise similarity matrix for positive pairs, shape [B, B]
            neg_sim: Pairwise similarity matrix for negative pairs, shape [B, B]
            labels: Label vector for samples at the current hierarchy level, shape [B]
            device: Device used for computations
            dtype: Data type
        Output:
            loss: loss of supervised contrastive learning loss (averaged across clusters)
        """
        masked_labels = labels.clone()
        masked_labels[:, level + 1:] = 0

        unique_combinations = torch.unique(masked_labels, dim=0)
        # uniq = torch.unique(labels[:,level])

        B = neg_sim.size(0)
        not_eye = ~torch.eye(B, dtype=torch.bool, device=device)

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        number_of_clusters = 0
        for label_set in unique_combinations:
            cluster_mask = torch.stack([torch.all(torch.eq(masked_labels[i], label_set))
                                        for i in range(masked_labels.shape[0])]).type(torch.uint8).to(device)  # [B]
            # cluster_mask = (labels == label)  # [B]
            if cluster_mask.sum() <= 1:
                # No positive pairs
                continue
            number_of_clusters += 1
            # Negative term
            neg_pair_mask = (cluster_mask[:, None] & (~cluster_mask)[None, :] & not_eye)
            neg_sim_masked = neg_sim[neg_pair_mask]
            neg_term = torch.logsumexp(neg_sim_masked, dim=0)

            # Positive term
            pos_pair_mask = cluster_mask[:, None] & cluster_mask[None, :]
            if self.is_symmetric:
                pos_pair_mask = torch.triu(pos_pair_mask, diagonal=1)
            else:
                pos_pair_mask &= not_eye
            pos_sim_masked = pos_sim[pos_pair_mask]
            pos_term = torch.logsumexp(-pos_sim_masked, dim=0)
            if self.is_symmetric:
                pos_term = pos_term * 2

            # Constant term
            c = cluster_mask.sum().to(dtype)  # |c_r|
            const_term = -2.0 * torch.log(c) - torch.log(c - 1.0)

            c_loss = neg_term + pos_term + const_term
            loss = loss + c_loss

        # num_clusters = unique.numel()
        if number_of_clusters > 0:
            loss = loss / number_of_clusters

        return loss

    def forward(self, features, labels):
        """
        Compute the hierarchical contrastive aggregation loss across multiple levels:
        The forward pass iterates from the finest level to the coarsest level,
        aggregating per-level supervised contrastive losses with exponential weighting.

        Input:

        Output:
        """
        # labels = labels[:,:2]
        batch_size = features.shape[0]
        num_sent = features.shape[1]
        L = labels.shape[1]  # finest level of hierarchy 3 is just image
        device = features.device
        dtype = features.dtype

        if num_sent == 2:
            z1, z2 = features[:, 0], features[:, 1]
        else:
            z1, z2 = features[:, 0], features[:, 0]

        # similarity matrix
        if self.augmentation:
            cross_view = torch.mm(z1, z2.t())
            sim = torch.cat([torch.mm(z1, z1.t()), cross_view], dim=1)  # [B, 2B]
            key_labels = torch.cat([labels, labels], dim=0)  # labels for keys
        else:
            sim = torch.mm(z1, z1.t())  # [B, B]
            key_labels = labels
        sim = sim / self.temperature

        not_eye = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        # Precompute neg term
        if self.clamp_negative:
            neg_sim = sim.clamp_min(0.0)
        else:
            neg_sim = sim

        def lambda_l(level, max_level=L):
            return torch.exp(torch.tensor(1.0 / (max_level - level), dtype=dtype, device=device))

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_lambda = torch.tensor(0.0, device=device, dtype=dtype)
        d_agg_next = torch.full((batch_size,), float('inf'), device=device, dtype=dtype)
        mask = torch.ones(labels.shape).to(device)
        # for level in range(L - 1, -1, -1):  # start from finer move to coarser
        # y = labels[:, level].to(device)
        for level in range(1, labels.shape[1]):
            mask[:, labels.shape[1] - level:] = 0
            layer_labels = labels * mask
            same_mask = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                     for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)  # matrix
            same_mask = same_mask & not_eye
            # prepare d_agg for the next iteration # okk hala sab kon
            # y = labels[:,0]
            # same_mask = (y[None, :] == y[:, None]) & not_eye  # [B,B]
            # aggregate per anchor i over its same-cluster members at level r (exclude self)
            if self.agg_type == "mean":
                # choose mean as aggregator
                sum_same = (sim * same_mask).sum(dim=1)
                cnt_same = torch.sum(same_mask, dim=1)
                cnt_same_clamp = same_mask.sum(dim=1).clamp_min(1)  # avoid div-by-zero
                # anchors with label 0 at this level don't form a cluster (set their d_agg to inf)
                d_agg_current = torch.where(cnt_same > 0,
                                            (sum_same / cnt_same_clamp).detach(),
                                            torch.full_like(sum_same, float('inf')))
            elif self.agg_type == "max":
                # choose max as aggregator
                max_same, _ = (sim * same_mask).max(dim=1)
                cnt_same = torch.sum(same_mask, dim=1)

                d_agg_current = torch.where(cnt_same > 0,
                                            max_same.detach(),
                                            torch.full_like(max_same, float('inf')))
            else:
                raise NotImplementedError(f"Aggregation type {self.agg_type} is not supported.")

            pos_sim = torch.minimum(d_agg_next, sim)  # [B,B]

            level_loss = self.supervised_contrastive_learning(pos_sim, neg_sim, labels, L - 1 - level, device, dtype)

            d_agg_next = d_agg_current

            total_loss = total_loss + lambda_l(L - 1 - level, L) * level_loss
            total_lambda = total_lambda + lambda_l(L - 1 - level, L)

        return total_loss / total_lambda  # , sim
