'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import io
import json
import math
import os
import random
import tarfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import glob
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Tar support
# ---------------------------------------------------------------------------

# Per-process cache of open tarfile handles.  Populated lazily inside worker
# processes (after fork) so each DataLoader worker gets independent handles.
_TAR_HANDLES: dict = {}


def _load_image(filename: str) -> Image.Image:
    """Load a PIL image from either a plain file path or a 'tarpath::member' key."""
    if '::' not in filename:
        return Image.open(filename).convert('RGB')
    tar_path, member_name = filename.split('::', 1)
    key = (os.getpid(), tar_path)
    if key not in _TAR_HANDLES:
        _TAR_HANDLES[key] = tarfile.open(tar_path, 'r:')
    tf = _TAR_HANDLES[key]
    f = tf.extractfile(tf.getmember(member_name))
    return Image.open(io.BytesIO(f.read())).convert('RGB')


def _index_one_tar(args):
    tar_path, cls = args
    with tarfile.open(tar_path, 'r:') as tf:
        return [(f'{tar_path}::{m.name}', cls)
                for m in tf.getmembers() if m.isfile()]


def _index_train_dir(root_dir: str, cache_path: str = None):
    """Scan root_dir/train/ for extracted class dirs or per-class tar files.

    Supports two layouts automatically:
      extracted:  root/train/{cls}/image.JPEG
      tar-based:  root/train/{cls}.tar   (one tar per class)

    An optional JSON cache file is written on first run and reloaded on
    subsequent runs so that indexing 1 000 tar files only happens once.

    Returns
    -------
    class_names : list[str]   sorted synset IDs (e.g. ['n01440764', ...])
    entries     : list[tuple] (filename_key, cls_str) pairs
    """
    # ------------------------------------------------------------------
    # Try cache first
    # ------------------------------------------------------------------
    if cache_path and os.path.exists(cache_path):
        print(f'Loading ImageNet train index from cache: {cache_path}')
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data['class_names'], [tuple(e) for e in data['entries']]

    train_dir = os.path.join(root_dir, 'train')

    # ------------------------------------------------------------------
    # Extracted directories layout
    # ------------------------------------------------------------------
    dirs = sorted(d for d in glob.glob(train_dir + '/*/') if os.path.isdir(d))
    if dirs:
        class_names = [d.rstrip('/').split('/')[-1] for d in dirs]
        entries = []
        for d, cls in zip(dirs, class_names):
            for f in sorted(glob.glob(os.path.join(d, '*'))):
                if os.path.isfile(f):
                    entries.append((f, cls))
        # No caching needed for extracted layout (glob is fast on next run)
        return class_names, entries

    # ------------------------------------------------------------------
    # Tar-file layout
    # ------------------------------------------------------------------
    tar_files = sorted(glob.glob(os.path.join(train_dir, '*.tar')))
    if not tar_files:
        raise FileNotFoundError(
            f"No extracted class directories or .tar files found in {train_dir}.\n"
            "Make sure --root-dir points to the ImageNet root and that "
            "train/ contains either class subdirectories or per-class .tar files."
        )

    class_names = [os.path.splitext(os.path.basename(t))[0] for t in tar_files]
    print(f'Indexing {len(tar_files)} tar files from {train_dir} '
          f'(this runs once; use --imagenet-index-cache to persist the result)...',
          flush=True)

    args = list(zip(tar_files, class_names))
    entries = []
    # Use threads: tarfile listing is I/O-bound, threads give real speedup
    with ThreadPoolExecutor(max_workers=min(16, len(tar_files))) as ex:
        for result in ex.map(_index_one_tar, args):
            entries.extend(result)

    print(f'Indexed {len(entries)} images across {len(class_names)} classes.',
          flush=True)

    # ------------------------------------------------------------------
    # Write cache
    # ------------------------------------------------------------------
    if cache_path:
        print(f'Saving index cache to {cache_path} ...')
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'class_names': class_names, 'entries': entries}, f)

    return class_names, entries


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class ImagenetHierarchihcalDataset(Dataset):
    """ImageNet hierarchical training dataset.

    Labels are [super_cat_int, cat_int, sample_idx] (3 columns) so HMLC
    produces 2 loss terms: super-category level and category level.

    Supports both extracted-directory and per-class .tar layouts in train/.
    The val/ split is always assumed to be extracted.
    """

    def __init__(self, hierarchy_file, root_dir, transform=None,
                 index_cache=None):
        """
        Args:
            hierarchy_file: JSON mapping synset_id -> supercategory string.
            root_dir: ImageNet root (contains train/ and val/).
            transform: callable; should be TwoCropTransform for training.
            index_cache: optional path to cache the tar index JSON so that
                         re-indexing is skipped on subsequent runs.
        """
        self.transform = transform

        with open(hierarchy_file, 'r') as f:
            sub_super = json.load(f)

        class_names, entries = _index_train_dir(root_dir, cache_path=index_cache)

        # Build class -> int and supercategory -> int mappings
        class_map_str_to_int = {cls: i for i, cls in enumerate(class_names)}
        super_class_map_str_to_int = {}
        super_cls_cnt = 0

        self.filenames = []
        self.category = []
        self.super_category = []
        # labels dict: {super_cls_int: {cls_int: [img_indices]}}
        self.labels = {}

        for filename_key, cls in entries:
            cls_int = class_map_str_to_int[cls]
            super_cls = sub_super.get(cls, cls)  # fallback to cls if not in hierarchy
            if super_cls not in super_class_map_str_to_int:
                super_class_map_str_to_int[super_cls] = super_cls_cnt
                super_cls_cnt += 1
            super_cls_int = super_class_map_str_to_int[super_cls]

            idx = len(self.filenames)
            self.filenames.append(filename_key)
            self.category.append(cls_int)
            self.super_category.append(super_cls_int)

            if super_cls_int not in self.labels:
                self.labels[super_cls_int] = {}
            if cls_int not in self.labels[super_cls_int]:
                self.labels[super_cls_int][cls_int] = []
            self.labels[super_cls_int][cls_int].append(idx)

    def get_label_split_by_index(self, index):
        return int(self.super_category[index]), int(self.category[index])

    def __getitem__(self, index):
        images0, images1, labels = [], [], []
        for i in index:
            image = _load_image(self.filenames[i])
            label = list(self.get_label_split_by_index(i)) + [i]
            if self.transform:
                image0, image1 = self.transform(image)
            images0.append(image0)
            images1.append(image1)
            labels.append(label)

        return [torch.stack(images0), torch.stack(images1)], torch.tensor(labels)

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
        # leaf nodes are lists of image indices
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


# ---------------------------------------------------------------------------
# Eval dataset  (val split — always extracted)
# ---------------------------------------------------------------------------

class ImagenetHierarchihcalDatasetEval(Dataset):
    """ImageNet evaluation dataset.

    Reads images from val/ (must be extracted into class subdirectories).
    Uses the train/ directory (extracted or tar) only to build a consistent
    class -> int mapping that matches ImagenetHierarchihcalDataset.

    Labels are [super_cat_int, cat_int].
    """

    def __init__(self, hierarchy_file, root_dir, transform=None,
                 index_cache=None):
        """
        Args:
            hierarchy_file: JSON mapping synset_id -> supercategory string.
            root_dir: ImageNet root (contains train/ and val/).
            transform: callable applied to each image.
            index_cache: same cache path used for the training dataset so that
                         class-name discovery reuses the cached index.
        """
        self.transform = transform

        with open(hierarchy_file, 'r') as f:
            sub_super = json.load(f)

        # Discover class names from train (consistent with training dataset)
        class_names, _ = _index_train_dir(root_dir, cache_path=index_cache)
        class_map_str_to_int = {cls: i for i, cls in enumerate(class_names)}

        super_class_map_str_to_int = {}
        super_cls_cnt = 0

        self.filenames = []
        self.category = []
        self.super_category = []
        self.labels = {}

        val_dir = os.path.join(root_dir, 'val')
        for cls in class_names:
            cls_int = class_map_str_to_int[cls]
            super_cls = sub_super.get(cls, cls)
            if super_cls not in super_class_map_str_to_int:
                super_class_map_str_to_int[super_cls] = super_cls_cnt
                super_cls_cnt += 1
            super_cls_int = super_class_map_str_to_int[super_cls]

            cls_val_dir = os.path.join(val_dir, cls)
            if not os.path.isdir(cls_val_dir):
                continue
            files = sorted(glob.glob(os.path.join(cls_val_dir, '*')))

            for filepath in files:
                if not os.path.isfile(filepath):
                    continue
                idx = len(self.filenames)
                self.filenames.append(filepath)
                self.category.append(cls_int)
                self.super_category.append(super_cls_int)

                if super_cls_int not in self.labels:
                    self.labels[super_cls_int] = {}
                if cls_int not in self.labels[super_cls_int]:
                    self.labels[super_cls_int][cls_int] = []
                self.labels[super_cls_int][cls_int].append(idx)

        self.targets = self.category.copy()

    def get_label_split_by_index(self, index):
        return int(self.super_category[index]), int(self.category[index])

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        label = list(self.get_label_split_by_index(index))
        if self.transform:
            image = self.transform(image)
        return image, label

    def random_sample(self, label, label_dict):
        curr_dict = label_dict
        top_level = True
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


# ---------------------------------------------------------------------------
# Hierarchical batch sampler
# ---------------------------------------------------------------------------

class HierarchicalBatchSampler(Sampler):
    """2-level hierarchical batch sampler for ImageNet (super_cls -> cls).

    For each anchor, samples one same-class image and one same-supercategory
    (different class) image, giving HMLC three contrastive levels per anchor.
    No distributed training requirement: defaults to num_replicas=1, rank=0.
    """

    def __init__(self, batch_size: int,
                 drop_last: bool, dataset: ImagenetHierarchihcalDataset,
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
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        print(self.total_size, self.num_replicas, self.batch_size,
              self.num_samples, len(self.dataset), self.rank)

    def random_unvisited_sample(self, label, label_dict, remaining, num_attempt=10):
        """Return a random index that is still in `remaining` (a set).

        Using a set for `remaining` gives O(1) membership tests, replacing
        the original O(N) list scan that caused OOM on large datasets.
        """
        for _ in range(num_attempt):
            idx = self.dataset.random_sample(label, label_dict)
            if idx in remaining:
                return idx
        # Fallback: return any unvisited index (remaining is a set, so
        # next(iter(...)) is O(1) and avoids an O(N) list conversion).
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

        # Maintain `remaining` as a set updated incrementally (O(1) discard).
        # The original code called list(set(indices).difference(visited)) TWICE
        # per batch iteration — O(N) allocations × N_batches caused gradual RSS
        # growth and eventually OOM-killed a DataLoader worker on ImageNet.
        remaining = set(indices)

        while len(remaining) > self.batch_size:
            # Pick a random anchor; skip if already consumed by a prior triplet
            idx = indices[torch.randint(len(indices), (1,)).item()]
            if idx not in remaining:
                continue
            remaining.discard(idx)
            batch.append(idx)
            super_cls, cls = self.dataset.get_label_split_by_index(idx)
            cls_index = self.random_unvisited_sample(
                cls, self.dataset.labels[super_cls], remaining)
            remaining.discard(cls_index)
            super_cls_index = self.random_unvisited_sample(
                super_cls, self.dataset.labels, remaining)
            remaining.discard(super_cls_index)
            batch.extend([super_cls_index, cls_index])
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