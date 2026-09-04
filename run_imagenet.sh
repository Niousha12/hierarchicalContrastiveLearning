#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=72:00:00
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

ENVPATH="venv"
source "$ENVPATH/bin/activate"
echo 'Venv activated'

cd /lustre06/project/6045013/nsadjadi/hierarchicalContrastiveLearning
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p slogs

# ---- Adjust these paths ----
IMAGENET_ROOT="/project/6045013/nsadjadi/data/imagenet"
HIERARCHY_FILE="$PWD/data_processing/imagenet_hierarchy.json"
# ----------------------------

python classification/train_imagenet.py \
    --root-dir "${IMAGENET_ROOT}" \
    --hierarchy-file "${HIERARCHY_FILE}" \
    --num-classes 1000 \
    --learning_rate 0.1 \
    --lr_decay_epochs '40,80' \
    --lr_decay_rate 0.1 \
    --temp 0.1 \
    --batch-size 512 \
    --epochs 100 \
    --criterion hmlc \
    --loss hmce \
    --model resnet50 \
    --workers 8 \
    --seed 0 \
    --tag imagenet \
    "$@"
# Note: --pretrained is NOT passed → trains from scratch (paper setting).
# To finetune from pretrained instead add: --pretrained --ckpt pretrained_model/resnet50-19c8e357.pth