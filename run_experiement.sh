#!/bin/bash

# ================= 實驗核心設定 (Core Settings) =================

# --- 模型設定 (Model Settings) ---
OPTIMIZER_MODEL="qwen2.5:32b"
SCORER_MODEL="qwen2.5:7b"

# --- 實驗參數 (Experiment Params) ---
# MMLU 子任務選擇
TARGET_SUBSETS="high_school_mathematics,high_school_chemistry,high_school_physics,high_school_world_history,business_ethics" 
# 每個子任務的訓練筆數 (用於快速實驗)
TRAIN_NUM=100
# 是否打亂訓練資料
SHUFFLE="true"

# --- GPO 優化參數 (GPO Hyperparameters) ---
# 搜尋輪數 (Epochs)
SEARCH_EPOCHS=2
# 每一步生成的 Prompt 數量 
GEN_NUM=5
# 評分器的溫度 (0.0 較為穩定)
SCORER_TEMP=0.0
# 優化器的溫度 (1.0 增加多樣性)
OPTIMIZER_TEMP=0.7

# 初始 Prompt
INITIAL_INSTRUCTION="Let's think step by step."
# Prompt 插入位置 (Q_end 代表問題之後)
INSTRUCTION_POS="Q_end"

# 批次大小
OPT_BATCH_SIZE=8
# Format Data 數量
FORMAT_DATA_NUM=3

# --- 動量與梯度設定 (Momentum & Gradient) ---
# GRADIENT_NAME="-" # 設為 "-" 代表不使用顯式梯度文字，或依程式邏輯調整
GRADIENT_NAME="feedback" 
MOMENTUM_PARA_NAME="para"
MOMENTUM_SELECTION_NAME="relavance"
MOMENTUM_SELECTION_NUM=3
MOMENTUM_UPDATE_NAME="k-list"

# --- 學習率與步長 (Learning Rate & Step Size) ---
LEARNING_RATE_NAME="w_lr"
UTIL_GRADIENT_NAME="generate"
# UTIL_GRADIENT_NAME="generate_without"
INITIAL_STEP_SIZE=50
DECAY_STRATEGY="consine"
USE_WARMUP_STRATEGY=false
WARMUP_STEPS=0
FINAL_STEP_SIZE=10

# 其他設定
INCLUDE_QA=false
GPUS="1"

# ================================================================

echo "----------------------------------------------------------------"
echo "執行 GPO 實驗:"
echo "Optimizer Model: $OPTIMIZER_MODEL"
echo "Scorer Model:    $SCORER_MODEL"
echo "Target Subsets:  $TARGET_SUBSETS"
echo "Train Num:       $TRAIN_NUM"
echo "Gen Num:         $GEN_NUM"
echo "----------------------------------------------------------------"

export PYTHONPATH=$PYTHONPATH:.

# 注意：雖然移除了 API Key 變數，但為了避免程式報錯，
# 下方指令仍保留參數欄位，但傳入空字串。

python src/optimization/main.py \
  --openai_api_key="" \
  --openai_api_key_list="" \
  --optimizer_llm_name "$OPTIMIZER_MODEL" \
  --scorer_llm_name "$SCORER_MODEL" \
  --optimizer_temperature $OPTIMIZER_TEMP \
  --scorer_temperature $SCORER_TEMP \
  --dataset mmlu \
  --mmlu_subsets $TARGET_SUBSETS \
  --mmlu_train_num $TRAIN_NUM \
  --shuffle_train_data $SHUFFLE \
  --num_search_epochs=$SEARCH_EPOCHS \
  --instruction_pos=$INSTRUCTION_POS \
  --initial_instruction="$INITIAL_INSTRUCTION" \
  --num_generated_instructions_in_each_step=$GEN_NUM \
  --opt_batch_size=$OPT_BATCH_SIZE \
  --format_data_num=$FORMAT_DATA_NUM \
  --gradient_name=$GRADIENT_NAME \
  --momentum_para_name=$MOMENTUM_PARA_NAME \
  --momentum_selection_name=$MOMENTUM_SELECTION_NAME \
  --momentum_selection_num=$MOMENTUM_SELECTION_NUM \
  --momentum_update_name=$MOMENTUM_UPDATE_NAME \
  --learning_rate_name=$LEARNING_RATE_NAME \
  --util_gradient_name=$UTIL_GRADIENT_NAME \
  --initial_step_size=$INITIAL_STEP_SIZE \
  --decay_strategy=$DECAY_STRATEGY \
  --use_warmup_strategy=$USE_WARMUP_STRATEGY \
  --warmup_steps=$WARMUP_STEPS \
  --final_step_size=$FINAL_STEP_SIZE \
  --include_qa=$INCLUDE_QA \
  --gpus=$GPUS