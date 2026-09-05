#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00
#SBATCH --output=slogs/%x__%A_%a.out
# Uncomment and set your ComputeCanada allocation account:
# #SBATCH --account=def-yourpi

# ---------------------------------------------------------------------------
# Data preparation (one-time, before first run):
#
#   1. Download iNaturalist 2018 from the official competition page:
#        https://github.com/visipedia/inat_comp/tree/master/2018
#
#      Files to download (~130 GB total):
#        train2018.tar.gz        (~120 GB)  training images
#        val2018.tar.gz          (~  9 GB)  validation images
#        train2018.json.tar.gz   annotation JSON for train split
#        val2018.json.tar.gz     annotation JSON for val split
#
#      Example (using wget or aria2c for faster download):
#        cd $INAT_ROOT
#        wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.tar.gz
#        wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.tar.gz
#        wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz
#        wget https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz
#        tar -xf train2018.tar.gz
#        tar -xf val2018.tar.gz
#        tar -xf train2018.json.tar.gz
#        tar -xf val2018.json.tar.gz
#
#   2. Expected directory structure after extraction:
#        $INAT_ROOT/
#          train2018.json
#          val2018.json
#          train_val2018/
#            Actinopterygii/
#              ...
#            Amphibia/
#              ...
#            (images are nested by kingdom/family/genus/species)
#
#   3. The genus is extracted automatically from the first word of each
#      scientific species name in the JSON. If you have a custom mapping
#      (category_id -> genus string) you can pass it via --hierarchy-file.
#
#   4. Download pretrained ResNet-50 weights (used by default):
#        mkdir -p pretrained_model
#        wget https://download.pytorch.org/models/resnet50-19c8e357.pth \
#             -P pretrained_model/
# ---------------------------------------------------------------------------

ENVPATH="venv"
source "$ENVPATH/bin/activate"
echo 'Venv activated'

cd /lustre06/project/6045013/nsadjadi/hierarchicalContrastiveLearning
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir -p slogs

# ---- Adjust these paths ----
INAT_ROOT="/scratch/nsadjadi/iNaturalist"
ANN_TRAIN="${INAT_ROOT}/train2018.json"
ANN_VAL="${INAT_ROOT}/val2018.json"
# Optional: path to a JSON file mapping category_id (int) -> genus string.
# Leave empty to use automatic first-word extraction from species names.
HIERARCHY_FILE=""
# ----------------------------

HIER_ARG=""
if [ -n "${HIERARCHY_FILE}" ]; then
    HIER_ARG="--hierarchy-file ${HIERARCHY_FILE}"
fi

python classification/train_inat.py \
    --root-dir "${INAT_ROOT}" \
    --ann-file-train "${ANN_TRAIN}" \
    --ann-file-val "${ANN_VAL}" \
    ${HIER_ARG} \
    --query-ratio 0.2 \
    --num-classes 8142 \
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
    --tag inat \
    "$@"
# Note: --pretrained is NOT passed → uses ImageNet pretrained weights by default
# (action='store_false' means passing --pretrained would DISABLE pretraining).
# Pretrained weights are loaded from pretrained_model/resnet50-19c8e357.pth
# unless --ckpt is specified.