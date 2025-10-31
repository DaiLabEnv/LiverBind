#!/bin/bash

GPU=0
TASK="regression"    # regression / classification
MODEL_PATH="./logs/models/best_model.pth"
TEST_DIR="dataset/test_index.csv"
SAVE_DIR="./logs/test_results"

CUDA_VISIBLE_DEVICES=${GPU} python3 ./main.py \
    --gpu ${GPU} \
    --task ${TASK} \
    --p_gcn --m_gcn \
    --resume \
    --pretrain_weights ${MODEL_PATH} \
    --test_dir ${TEST_DIR} \
    --save_dir ${SAVE_DIR}