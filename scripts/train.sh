#!/bin/bash

GPU=0
TASK="regression"   # regression / classification
BATCH_SIZE=32
EPOCHS=300
LR=1e-4
SAVE_DIR="./logs"
TRAIN_DIR="dataset/train_index.csv"

CUDA_VISIBLE_DEVICES=${GPU} python3 ./main.py \
    --gpu ${GPU} \
    --task ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --nepoch ${EPOCHS} \
    --lr_initial ${LR} \
    --save_dir ${SAVE_DIR} \
    --train_dir ${TRAIN_DIR} \
    --p_gcn --m_gcn