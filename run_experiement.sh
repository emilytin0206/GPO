#!/bin/bash

# ================= 實驗參數設定 =================

MODEL_NAME="llama3"
TARGET_SUBSETS="abstract_algebra anatomy"
TRAIN_NUM=5
SHUFFLE="true"

# 新增 Scorer Temperature
SCORER_TEMP=0.0

SEARCH_EPOCHS=2
GEN_NUM=4

# ==============================================

echo "----------------------------------------------------------------"
echo "執行設定:"
echo "模型: $MODEL_NAME"
echo "Scorer Temperature: $SCORER_TEMP"
echo "----------------------------------------------------------------"

export PYTHONPATH=$PYTHONPATH:.

python src/optimization/main.py \
    --dataset mmlu \
    --optimizer_llm_name "$MODEL_NAME" \
    --scorer_llm_name "$MODEL_NAME" \
    --scorer_temperature $SCORER_TEMP \
    --mmlu_subsets $TARGET_SUBSETS \
    --mmlu_train_num $TRAIN_NUM \
    --shuffle_train_data $SHUFFLE \
    --num_search_epochs $SEARCH_EPOCHS \
    --num_generated_instructions_in_each_step $GEN_NUM \
    --gpus 0