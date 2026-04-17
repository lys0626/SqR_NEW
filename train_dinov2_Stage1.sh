#!/bin/bash
# [新增] 强行让系统优先使用 lys2 虚拟环境里的 C++ 库！
set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=3
# export CUDA_VISIBLE_DEVICES=${GPU_ID}      # 指定使用的单卡 GPU 编号

DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径

# DATASET_NAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"

EXP_DIR="./experiment/4_17_0.975_DinoV2_ASL_AdamW_1e-4_sample"        # 实验输出的顶层根目录
# EXP_DIR="./experiment/vision"
# ================= 2. 方法选择配置 =================
# 可选值: "splicemix" 或 "splicemix-cl"，或 baseline（不使用任何增强）
STAGE2_METHOD="splicemix-cl"             # 配置为 "splicemix" 或 "splicemix-cl" 或 baseline
NUM_CLASS=13

# ================= 3. 动态生成输出目录 =================
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}/stage1_${STAGE2_METHOD}_q2l"
STAGE2_OUT="${EXP_DIR}/${DATASET_NAME}/stage2_${STAGE2_METHOD}"

mkdir -p "${STAGE1_OUT}"
mkdir -p "${STAGE2_OUT}"

echo "==================================================="
echo "  [STEP 1] 启动 Stage 1: Q2L 预热与 MEE(Early Cutting) 标签级清洗"
echo "  ==> 采用算法: ICCV2023 LateStopping + NeurIPS2025 MEE"
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="

python stage1_main_dinov2.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-4 \
  -cd ${GPU_ID} \
  --img_size 224 \
  --num_class ${NUM_CLASS} \
  --workers 8 \
  --pretrained \
  --warm_up_epochs 6 \
  --fkl_consecutive_epochs 3 \
  --early_cutting_rate 1.5 \
  --newremove_rate 75000 \
  --top_conf_ratio 0.2 \
  --low_grad_ratio 0.2 \
  --optim AdamW 
