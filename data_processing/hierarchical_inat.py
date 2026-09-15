'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import os
import json
import torch
import math
from typing import Optional
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from PIL import Image
import random


class INatHierarchicalDataset(Dataset):
    """iNaturalist hierarchical training dataset.

    Labels are ordered [genus_int, species_int, sample_idx] so HMLC produces
    2 loss terms: genus-level and species-level (sample_idx gives SimCLR-like
    finest-level behaviour where only the two TwoCropTransform views are positive).
    """

    def __init__(self, root_dir, ann_file, transform=None, hierarchy_file=None):
        """
        Args:
            root_dir: root directory containing iNaturalist images.
            ann_file: path to iNat JSON annotation file with keys
                      'images', 'annotations', 'categories'.
            transform: callable transform applied to each image (should be
                       TwoCropTransform for training).
            hierarchy_file: optional JSON mapping category_id (int or str) to
                            genus string. If None, genus is extracted from the
                            first word of category['name'] (scientific naming).
        """
        self.root_dir = root_dir
        self.transform = transform

        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Build category_id -> category info lookup
        cat_info = {}
        for cat in data['categories']:
            cat_info[cat['id']] = cat

        # Optional external genus mapping: category_id -> genus string
        genus_map = {}
        if hierarchy_file is not None:
            with open(hierarchy_file, 'r') as f:
                raw = json.load(f)
            # Keys may be stored as strings in JSON
            for k, v in raw.items():
                genus_map[int(k)] = v

        # Build image_id -> file_name lookup
        img_lookup = {}
        for img in data['images']:
            img_lookup[img['id']] = img['file_name']

        # Build deterministic mappings from sorted categories so that
        # the same cat_id always gets the same integer regardless of which
        # annotation file is used.
        sorted_cats = sorted(data['categories'], key=lambda c: c['id'])

        genus_str_to_int = {}
        genus_cnt = 0
        for cat in sorted_cats:
            cid = cat['id']
            if genus_map:
                g = genus_map.get(cid, cat.get('genus', str(cid)))
            else:
                g = cat.get('genus', str(cid))
            if g not in genus_str_to_int:
                genus_str_to_int[g] = genus_cnt
                genus_cnt += 1

        species_str_to_int = {str(cat['id']): i for i, cat in enumerate(sorted_cats)}

        self.filenames = []
        self.genus_labels = []
        self.species_labels = []
        # labels dict: {genus_int: {species_int: [img_indices]}}
        self.labels = {}

        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            file_name = img_lookup[img_id]

            cat = cat_info[cat_id]
            species_str = str(cat_id)

            # Determine genus: priority order:
            #   1. external hierarchy_file mapping
            #   2. 'genus' field in the category JSON (iNat 2018 has this)
            #   3. cat_id string as fallback
            if genus_map:
                genus_str = genus_map.get(cat_id, cat.get('genus', str(cat_id)))
            else:
                genus_str = cat.get('genus', str(cat_id))

            genus_int = genus_str_to_int[genus_str]
            species_int = species_str_to_int[species_str]

            idx = len(self.filenames)
            self.filenames.append(os.path.join(root_dir, file_name))
            self.genus_labels.append(genus_int)
            self.species_labels.append(species_int)

            if genus_int not in self.labels:
                self.labels[genus_int] = {}
            if species_int not in self.labels[genus_int]:
                self.labels[genus_int][species_int] = []
            self.labels[genus_int][species_int].append(idx)

    def get_label_split_by_index(self, index):
        return int(self.genus_labels[index]), int(self.species_labels[index])

    def __getitem__(self, index):
        images0, images1, labels = [], [], []
        for i in index:
            image = Image.open(self.filenames[i]).convert('RGB')
            genus_int, species_int = self.get_label_split_by_index(i)
            label = [genus_int, species_int, i]
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


class INatHierarchicalDatasetEval(Dataset):
    """iNaturalist evaluation dataset.

    Labels are [genus_int, species_int]. Used for memory/query/gallery splits.
    __getitem__ is called with a single integer index by the standard DataLoader.
    """

    def __init__(self, root_dir, ann_file, transform=None, hierarchy_file=None):
        """
        Args:
            root_dir: root directory containing iNaturalist images.
            ann_file: path to iNat JSON annotation file.
            transform: callable transform applied to each image.
            hierarchy_file: optional JSON mapping category_id to genus string.
        """
        self.root_dir = root_dir
        self.transform = transform

        with open(ann_file, 'r') as f:
            data = json.load(f)

        cat_info = {}
        for cat in data['categories']:
            cat_info[cat['id']] = cat

        genus_map = {}
        if hierarchy_file is not None:
            with open(hierarchy_file, 'r') as f:
                raw = json.load(f)
            for k, v in raw.items():
                genus_map[int(k)] = v

        img_lookup = {}
        for img in data['images']:
            img_lookup[img['id']] = img['file_name']

        # Build deterministic mappings from sorted categories so that
        # the same cat_id always gets the same integer regardless of which
        # annotation file (train vs val) is used. This is critical for KNN
        # evaluation where the memory bank (train) and queries (val) must
        # share the same label space.
        sorted_cats = sorted(data['categories'], key=lambda c: c['id'])

        genus_str_to_int = {}
        genus_cnt = 0
        for cat in sorted_cats:
            cid = cat['id']
            if genus_map:
                g = genus_map.get(cid, cat.get('genus', str(cid)))
            else:
                g = cat.get('genus', str(cid))
            if g not in genus_str_to_int:
                genus_str_to_int[g] = genus_cnt
                genus_cnt += 1

        species_str_to_int = {str(cat['id']): i for i, cat in enumerate(sorted_cats)}

        self.filenames = []
        self.genus_labels = []
        self.species_labels = []
        self.labels = {}

        for ann in data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            file_name = img_lookup[img_id]

            cat = cat_info[cat_id]
            species_str = str(cat_id)

            if genus_map:
                genus_str = genus_map.get(cat_id, cat.get('genus', str(cat_id)))
            else:
                genus_str = cat.get('genus', str(cat_id))

            genus_int = genus_str_to_int[genus_str]
            species_int = species_str_to_int[species_str]

            idx = len(self.filenames)
            self.filenames.append(os.path.join(root_dir, file_name))
            self.genus_labels.append(genus_int)
            self.species_labels.append(species_int)

            if genus_int not in self.labels:
                self.labels[genus_int] = {}
            if species_int not in self.labels[genus_int]:
                self.labels[genus_int][species_int] = []
            self.labels[genus_int][species_int].append(idx)

        self.targets = self.species_labels.copy()

    def get_label_split_by_index(self, index):
        return int(self.genus_labels[index]), int(self.species_labels[index])

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        genus_int, species_int = self.get_label_split_by_index(index)
        label = [genus_int, species_int]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.filenames)


class HierarchicalBatchSampler(Sampler):
    """2-level hierarchical batch sampler for iNaturalist (genus -> species).

    For each anchor, samples one same-species image and one same-genus
    (different species) image. Batches are groups of
    (anchor, same_species_idx, same_genus_idx) yielded when
    len(batch) >= batch_size.

    No distributed training requirement: defaults to num_replicas=1, rank=0.
    """

    def __init__(self, batch_size: int,
                 drop_last: bool, dataset: INatHierarchicalDataset,
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

    def random_unvisited_sample(self, label, label_dict, remaining, num_attempt=10):
        """Return a random index that is still in `remaining` (a set)."""
        for _ in range(num_attempt):
            idx = self.dataset.random_sample(label, label_dict)
            if idx in remaining:
                return idx
        return next(iter(remaining))

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        batch = []
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        if not self.drop_last:
            indices += indices[:(self.total_size - len(indices))]
        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Maintain `remaining` as an incrementally-updated set (O(1) discard)
        # instead of recomputing list(set(indices).difference(visited)) each
        # batch — the original caused OOM on large datasets.
        remaining = set(indices)

        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,)).item()]
            if idx not in remaining:
                continue
            remaining.discard(idx)
            batch.append(idx)
            genus_int, species_int = self.dataset.get_label_split_by_index(idx)
            species_index = self.random_unvisited_sample(
                species_int, self.dataset.labels[genus_int], remaining)
            remaining.discard(species_index)
            genus_index = self.random_unvisited_sample(
                genus_int, self.dataset.labels, remaining)
            remaining.discard(genus_index)
            batch.extend([species_index, genus_index])
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if (len(remaining) > self.batch_size) and not self.drop_last:
            batch.update(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
