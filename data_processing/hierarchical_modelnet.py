'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import glob
import torch
import math
from typing import Optional
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from PIL import Image
import random


def _find_images(directory):
    """Return sorted list of image paths under directory using flexible extension search."""
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        files = sorted(glob.glob(os.path.join(directory, ext)))
        if files:
            return files
    return []


def _resolve_class_dirs(root_dir, split):
    """Return (layout, list of (cls_name, image_dir)) sorted by cls_name.

    Supports two directory layouts automatically:
      split_first:  root/{split}/{class}/images   (e.g. root/train/airplane/*.png)
      class_first:  root/{class}/{split}/images   (e.g. root/airplane/train/*.png)
    """
    split_first_dir = os.path.join(root_dir, split)
    if os.path.isdir(split_first_dir):
        # split_first layout
        class_names = sorted(
            d for d in os.listdir(split_first_dir)
            if os.path.isdir(os.path.join(split_first_dir, d))
        )
        return [(cls, os.path.join(split_first_dir, cls)) for cls in class_names]

    # class_first layout: root/{class}/{split}/
    class_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d, split))
    )
    if not class_names:
        raise FileNotFoundError(
            f"Could not find images in either '{split_first_dir}' (split-first layout) "
            f"or '{root_dir}/{{class}}/{split}/' (class-first layout). "
            f"Check that --root-dir is correct."
        )
    return [(cls, os.path.join(root_dir, cls, split)) for cls in class_names]


def _parse_model_id(filename):
    """Extract model_id from a ModelNet rendered-view filename.

    Expected pattern: {class_name}_{model_id:04d}_{view:03d}.{ext}
    The model_id is the token between the first and second-to-last underscore.
    If there is only one underscore the entire numeric suffix (before extension)
    is treated as the model_id.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    if len(parts) >= 3:
        # parts[-1] is view index, parts[-2] is model_id
        try:
            return int(parts[-2])
        except ValueError:
            pass
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return 0


class ModelNetHierarchicalDataset(Dataset):
    """ModelNet40 rendered-image hierarchical training dataset.

    Labels are ordered [cat_int, model_id_int, sample_idx] so HMLC produces
    2 loss terms: category-level and model-level (sample_idx gives SimCLR-like
    finest-level behaviour where only the two TwoCropTransform views are positive).

    Supported directory layouts (auto-detected):
      split_first:  {root}/{split}/{class_name}/*.{ext}
      class_first:  {root}/{class_name}/{split}/*.{ext}  <- modelnet40_images_new_12x
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: root directory of rendered ModelNet40 images.
            split: split subdirectory name, e.g. 'train' or 'test'.
            transform: callable transform (should be TwoCropTransform for training).
        """
        self.root_dir = root_dir
        self.transform = transform

        class_dirs = _resolve_class_dirs(root_dir, split)
        class_to_int = {cls: i for i, (cls, _) in enumerate(class_dirs)}

        self.filenames = []
        self.cat_labels = []
        self.model_id_labels = []
        # labels dict: {cat_int: {model_id_int: [img_indices]}}
        self.labels = {}

        # We need a global model_id integer mapping across the whole dataset so
        # that each unique physical model gets a unique integer.
        model_id_str_to_int = {}
        model_id_cnt = 0

        for cls_name, cls_dir in class_dirs:
            cat_int = class_to_int[cls_name]
            files = _find_images(cls_dir)

            for filepath in files:
                raw_model_id = _parse_model_id(filepath)
                # Make model_id unique per class by using (cat_int, raw_model_id)
                key = (cat_int, raw_model_id)
                if key not in model_id_str_to_int:
                    model_id_str_to_int[key] = model_id_cnt
                    model_id_cnt += 1
                model_id_int = model_id_str_to_int[key]

                idx = len(self.filenames)
                self.filenames.append(filepath)
                self.cat_labels.append(cat_int)
                self.model_id_labels.append(model_id_int)

                if cat_int not in self.labels:
                    self.labels[cat_int] = {}
                if model_id_int not in self.labels[cat_int]:
                    self.labels[cat_int][model_id_int] = []
                self.labels[cat_int][model_id_int].append(idx)

    def get_label_split_by_index(self, index):
        return int(self.cat_labels[index]), int(self.model_id_labels[index])

    def __getitem__(self, index):
        images0, images1, labels = [], [], []
        for i in index:
            image = Image.open(self.filenames[i]).convert('RGB')
            cat_int, model_id_int = self.get_label_split_by_index(i)
            label = [cat_int, model_id_int, i]
            if self.transform:
                image0, image1 = self.transform(image)
            images0.append(image0)
            images1.append(image1)
            labels.append(label)

        return [torch.stack(images0), torch.stack(images1)], torch.tensor(labels)

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        # all sub trees end with a list of indices
        while type(curr_dict) is not list:
            if top_level:
                random_label = label
                if len(curr_dict.keys()) != 1:
                    while random_label == label:
                        random_label = random.sample(list(curr_dict.keys()), 1)[0]
            else:
                random_label = random.sample(list(curr_dict.keys()), 1)[0]
            curr_dict = curr_dict[random_label]
            top_level = False
        return random.sample(curr_dict, 1)[0]

    def __len__(self):
        return len(self.filenames)


class ModelNetHierarchicalDatasetEval(Dataset):
    """ModelNet40 evaluation dataset.

    Labels are [cat_int, model_id_int]. Used for memory/test splits.
    __getitem__ is called with a single integer index by the standard DataLoader.
    """

    def __init__(self, root_dir, split='test', transform=None):
        """
        Args:
            root_dir: root directory of rendered ModelNet40 images.
            split: split subdirectory name, e.g. 'train' or 'test'.
            transform: callable transform applied to each image.
        """
        self.root_dir = root_dir
        self.transform = transform

        class_dirs = _resolve_class_dirs(root_dir, split)
        class_to_int = {cls: i for i, (cls, _) in enumerate(class_dirs)}

        self.filenames = []
        self.cat_labels = []
        self.model_id_labels = []
        self.labels = {}

        model_id_str_to_int = {}
        model_id_cnt = 0

        for cls_name, cls_dir in class_dirs:
            cat_int = class_to_int[cls_name]
            files = _find_images(cls_dir)

            for filepath in files:
                raw_model_id = _parse_model_id(filepath)
                key = (cat_int, raw_model_id)
                if key not in model_id_str_to_int:
                    model_id_str_to_int[key] = model_id_cnt
                    model_id_cnt += 1
                model_id_int = model_id_str_to_int[key]

                idx = len(self.filenames)
                self.filenames.append(filepath)
                self.cat_labels.append(cat_int)
                self.model_id_labels.append(model_id_int)

                if cat_int not in self.labels:
                    self.labels[cat_int] = {}
                if model_id_int not in self.labels[cat_int]:
                    self.labels[cat_int][model_id_int] = []
                self.labels[cat_int][model_id_int].append(idx)

        self.targets = self.cat_labels.copy()

    def get_label_split_by_index(self, index):
        return int(self.cat_labels[index]), int(self.model_id_labels[index])

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        cat_int, model_id_int = self.get_label_split_by_index(index)
        label = [cat_int, model_id_int]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.filenames)


class HierarchicalBatchSampler(Sampler):
    """2-level hierarchical batch sampler for ModelNet40 (category -> model_id).

    For each anchor, samples one same-model_id image and one same-category
    (different model_id) image. Batches are groups of
    (anchor, same_model_idx, same_cat_idx) yielded when
    len(batch) >= batch_size.

    No distributed training requirement: defaults to num_replicas=1, rank=0.
    """

    def __init__(self, batch_size: int,
                 drop_last: bool, dataset: ModelNetHierarchicalDataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:

        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch = 0
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank)

    def random_unvisited_sample(self, label, label_dict, visited, indices, remaining, num_attempt=10):
        attempt = 0
        while attempt < num_attempt:
            idx = self.dataset.random_sample(label, label_dict)
            if idx not in visited and idx in indices:
                visited.add(idx)
                return idx
            attempt += 1
        idx = remaining[torch.randint(len(remaining), (1,))]
        visited.add(idx)
        return idx

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        remaining = list(set(indices).difference(visited))
        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            cat_int, model_id_int = self.dataset.get_label_split_by_index(idx)
            # Sample a same-model_id image (different view/index)
            model_index = self.random_unvisited_sample(
                model_id_int, self.dataset.labels[cat_int], visited, indices, remaining)
            # Sample a same-category, different-model image
            cat_index = self.random_unvisited_sample(
                cat_int, self.dataset.labels, visited, indices, remaining)
            batch.extend([model_index, cat_index])
            visited.update([model_index, cat_index])
            remaining = list(set(indices).difference(visited))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
            remaining = list(set(indices).difference(visited))

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
