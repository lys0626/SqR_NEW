#!/bin/bash
# [新增] 强行让系统优先使用 lys2 虚拟环境里的 C++ 库！
set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=1
# export CUDA_VISIBLE_DEVICES=${GPU_ID}      # 指定使用的单卡 GPU 编号

# DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
# DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
# DATA_DIR="/data/mimic_cxr/PA_NEW"      # 数据集的根目录路径

# DATASET_NAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"
DATASET_NAME="vinbigdata"                     # 对应 stage1 里的 args.dataname
DATASET_NAME_UPPER="VINBIGDATA"               
DATA_DIR="/data/dsj/lys/vinbigdata"

# DATASET_NAME="chexpert"
# DATASET_NAME_UPPER="CHEXPERT"
# DATA_DIR="/data/chexpert_224"

EXP_DIR="./experiment/VINVIG_denoise/5_8_0.95_4_sym_0.2"        # 实验输出的顶层根目录
# EXP_DIR="./experiment/vision"
# ================= 2. 方法选择配置 =================
# 可选值: "splicemix" 或 "splicemix-cl"，或 baseline（不使用任何增强）
STAGE2_METHOD="splicemix-cl"             # 配置为 "splicemix" 或 "splicemix-cl" 或 baseline
NUM_CLASS=15

# ================= 3. 动态生成输出目录 =================
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}/stage1_${STAGE2_METHOD}"

mkdir -p "${STAGE1_OUT}"

echo "==================================================="
echo "  [STEP 1] 启动 Stage 1: Q2L 预热与 MEE(Early Cutting) 标签级清洗"
echo "  ==> 采用算法: ICCV2023 LateStopping + NeurIPS2025 MEE"
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="
  # --noise_type asym \
  # --fn_rate 0.5 \
  # --fp_rate 0.5 \
  # --noise_type sym \
  # --sym_rate 0.2
python stage1_main_resnet50.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-4 \
  -cd ${GPU_ID} \
  --img_size 224 \
  --num_class 15 \
  --fkl_consecutive_epochs 8 \
  --optim AdamW \
  --early_cutting_rate 5 \
  --warm_up_epochs 0 \
  --inject_noise \
  --noise_type sym \
  --sym_rate 0.2
