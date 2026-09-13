#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:a100:4       # 4 × 40 GB A100s — Narval cluster
# #SBATCH --gres=gpu:v100l:4    # 4 × 32 GB V100s — Cedar cluster (uncomment if on Cedar)
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=3-72:00:00
#SBATCH --output=slogs/%x__%A_%a.out
# Uncomment and set your ComputeCanada allocation account:
# #SBATCH --account=def-yourpi

# ---------------------------------------------------------------------------
# Data preparation (one-time, before first run):
#
#   ImageNet ILSVRC-2012 is available as a shared dataset on ComputeCanada.
#   On most clusters it lives at one of:
#     /project/rrg-*/data/imagenet/           (Graham / Cedar)
#     /project/def-*/datasets/ILSVRC2012/     (Narval / Beluga)
#   If it is not pre-installed, download from https://image-net.org (account
#   required) and unpack so the directory has the structure:
#
#     $IMAGENET_ROOT/
#       train/
#         n01440764/   <- synset folders
#         n01443537/
#         ...
#       val/
#         n01440764/
#         ...
#
#   Generate the supercategory hierarchy JSON once (needs NLTK + WordNet):
#
#     python scripts/prepare_imagenet_hierarchy.py \
#         --imagenet-root $IMAGENET_ROOT \
#         --output data_processing/imagenet_hierarchy.json
#
#   Download pretrained ResNet-50 weights (not needed for ImageNet: trained
#   from scratch, but needed if you ever want --pretrained):
#     wget https://download.pytorch.org/models/resnet50-19c8e357.pth \
#          -P pretrained_model/
# ---------------------------------------------------------------------------

ENVPATH="/home/nsadjadi/projects/def-lila-ab/nsadjadi/HiCAggLoss/hierarchical_contrastive/env"
source "$ENVPATH/bin/activate"
echo 'Venv activated'

cd /lustre06/project/6045013/nsadjadi/hierarchicalContrastiveLearning
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p slogs

# ---- Adjust these paths ----
IMAGENET_ROOT="/scratch/nsadjadi/imagenet"
HIERARCHY_FILE="$PWD/data_processing/imagenet_hierarchy.json"
# Path where the tar-member index is cached after the first run.
# Delete this file if you move/re-download the dataset.
INDEX_CACHE="$PWD/data_processing/imagenet_train_index.json"
# Number of GPUs per node (must match --gres=gpu:a100:N above)
GPUS_PER_NODE=4
# ----------------------------

# torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
# automatically for each spawned process.
#
# --batch-size is per-GPU; effective batch = batch_size × GPUS_PER_NODE.
# With 4 GPUs and batch-size 128 → effective batch 512.
torchrun --nproc_per_node=${GPUS_PER_NODE} \
    classification/train_imagenet.py \
    --root-dir "${IMAGENET_ROOT}" \
    --hierarchy-file "${HIERARCHY_FILE}" \
    --imagenet-index-cache "${INDEX_CACHE}" \
    --num-classes 1000 \
    --learning_rate 0.1 \
    --lr_decay_epochs '40,80' \
    --lr_decay_rate 0.1 \
    --temp 0.1 \
    --batch-size 128 \
    --epochs 100 \
    --criterion hmlc \
    --loss hmce \
    --model resnet50 \
    --workers 8 \
    --seed 0 \
    --tag imagenet \
    --amp \
    --eval-freq 20 \
    "$@"
# Note: --pretrained is NOT passed → trains from scratch (paper setting).
# To finetune from pretrained instead add: --pretrained --ckpt pretrained_model/resnet50-19c8e357.pth
#
# To run on a single GPU (no distribution), use:
#   torchrun --nproc_per_node=1 classification/train_imagenet.py ...