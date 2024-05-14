#!/bin/bash

MODEL="xlm-roberta-base"
GPU=${2:-6}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL_TYPE="xlmr"

OUTPUT_DIR="/mounts/data/proj/yihong/newhome/Transli-OFA/evaluation/sib200/results/"
DATA_DIR="/mounts/data/proj/yihong/datasets/sib-200/data/annotated"
init_checkpoint="/mounts/data/proj/yihong/newhome/Transli-OFA/saved_models"


python -u evaluate_sib_all.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --epochs 40 \
    --nr_of_seeds 5 \
    --init_checkpoint $init_checkpoint \
    --data_dir $DATA_DIR \
    --source_language "eng_Latn"