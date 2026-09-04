#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=slogs/%x__%A_%a.out
# Uncomment and set your ComputeCanada allocation account:
# #SBATCH --account=def-yourpi

# ---------------------------------------------------------------------------
# Data preparation (one-time, before first run):
#
#   ModelNet40 rendered 2D images (multi-view).
#
#   Option A — ModelNet40-12view (most common, used in MVCNN papers):
#     Download the 12-view rendered PNG images from:
#       https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v1png_norm12.tar
#     OR search for "ModelNet40 rendered images" / "modelnet40_images_new_12x"
#     on academic hosting sites (e.g., ModelNet40 from Point Cloud Library).
#
#   Option B — Render your own using the Princeton ModelNet tools:
#     https://modelnet.cs.princeton.edu/  (download .off mesh files)
#     Then render with Blender or the provided rendering scripts.
#
#   Expected directory structure (the code scans alphabetically sorted class dirs):
#
#     $MODELNET_ROOT/
#       train/
#         airplane/
#           airplane_0001_001.png   <- {class}_{model_id:04d}_{view:03d}.png
#           airplane_0001_002.png
#           ...
#           airplane_0002_001.png
#         bathtub/
#           ...
#         (40 class folders)
#       test/
#         airplane/
#           ...
#         (same 40 class folders)
#
#   The code also accepts .jpg and .jpeg extensions. model_id is parsed from
#   the second-to-last underscore-separated token in the filename.
#   Single-view datasets (one file per model) work too: model_id is the last
#   numeric token before the extension.
#
#   Download pretrained ResNet-50 weights (used by default):
#     mkdir -p pretrained_model
#     wget https://download.pytorch.org/models/resnet50-19c8e357.pth \
#          -P pretrained_model/
# ---------------------------------------------------------------------------

ENVPATH="venv"
source "$ENVPATH/bin/activate"
echo 'Venv activated'

cd /lustre06/project/6045013/nsadjadi/hierarchicalContrastiveLearning
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p slogs

# ---- Adjust this path ----
MODELNET_ROOT="/scratch/nsadjadi/modelNet40/modelnet40_images_new_12x"
# --------------------------

python classification/train_modelnet.py \
    --root-dir "${MODELNET_ROOT}" \
    --num-classes 40 \
    --learning_rate 0.1 \
    --lr_decay_epochs '40,80' \
    --lr_decay_rate 0.1 \
    --temp 0.1 \
    --batch-size 512 \
    --epochs 100 \
    --criterion hmlc \
    --loss hmce \
    --model resnet50 \
    --workers 4 \
    --seed 0 \
    --tag modelnet40 \
    "$@"
# Note: --pretrained is NOT passed → uses ImageNet pretrained weights by default.
# Pretrained weights are loaded from pretrained_model/resnet50-19c8e357.pth
# unless --ckpt is specified.