#!/bin/bash
# [新增] 强行让系统优先使用 lys2 虚拟环境里的 C++ 库！
set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=5
# export CUDA_VISIBLE_DEVICES=${GPU_ID}      # 指定使用的单卡 GPU 编号

# DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
# DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
# DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径

DATASET_NAME="nih"
DATASET_NAME_UPPER="NIH-CHEST"
DATA_DIR="/data/nih-chest-xrays"

EXP_DIR="./experiment/center_2"        # 实验输出的顶层根目录
# EXP_DIR="./experiment/vision"
# ================= 2. 方法选择配置 =================
# 可选值: "splicemix" 或 "splicemix-cl"，或 baseline（不使用任何增强）
STAGE2_METHOD="splicemix-cl"             # 配置为 "splicemix" 或 "splicemix-cl" 或 baseline
NUM_CLASS=14

# ================= 3. 动态生成输出目录 =================
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}/stage1_${STAGE2_METHOD}_q2l"
STAGE2_OUT="${EXP_DIR}/${DATASET_NAME}/stage2_${STAGE2_METHOD}"

mkdir -p "${STAGE1_OUT}"
mkdir -p "${STAGE2_OUT}"

echo "==================================================="
echo "  [STEP 1] 启动 Stage 1: Q2L 预热与 RoLT 数据清洗    "
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="

python stage1_main.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --epochs 16 \
  --splicemix_start_epoch 15 \
  --rolt_start_epoch 10 \
  --batch-size 32 \
  --optim AdamW \
  --lr 1e-4 \
  -cd ${GPU_ID} \
  --img_size 448 \
  --num_class ${NUM_CLASS} \
  --backbone resnet50 \
  --workers 16 \
  --pretrained \
  --momentum 0.9 \
  --keep_input_proj \
  --hidden_dim 512 \
  --dim_feedforward 2048 \
  --dec_layers 2 \
  --enc_layers 1 \
  --nheads 4 \
  --scheduler OneCycle
