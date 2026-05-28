#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=23:00:00
#SBATCH --output=slogs/%x__%A_%a.out

PROJECT_NAME="classification"
PROJECT_DIRN="${PROJECT_NAME//-/_}"

ENVPATH="venv"
source "$ENVPATH/bin/activate"
echo 'Venv activated'

cd /lustre06/project/6045013/nsadjadi/hierarchicalContrastiveLearning
export PYTHONPATH="$PWD:$PYTHONPATH"

python classification/train_deepfashion.py \
    --data deepfashion/ \
    --train-listfile train_listfile.json \
    --val-listfile val_listfile.json \
    --test-listfile test_listfile.json \
    --class-map-file class_map.json \
    --repeating-product-file repeating_product_ids.csv \
    --num-classes 17 \
    --learning_rate 0.05 \
    --temp 0.2 \
    --ckpt pretrained_model/ \
    --criterion hsmc \
    --dist-url 'tcp://localhost:10001' \
    --world-size 1 \
    --rank 0 \
    --cosine