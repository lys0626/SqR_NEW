#!/bin/bash
# [新增] 强行让系统优先使用 lys2 虚拟环境里的 C++ 库！
set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=6
# export CUDA_VISIBLE_DEVICES=${GPU_ID}      # 指定使用的单卡 GPU 编号

# DATASET_NAME="mimic"
# DATASET_NAME_UPPER="MIMIC"
# DATA_DIR="/data/mimic_cxr/PA/7_1_2"

# DATASET_NAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"

# DATASET_NAME="vinbigdata"                     # 对应 stage1 里的 args.dataname
# DATASET_NAME_UPPER="VINBIGDATA"               
# DATA_DIR="/data/dsj/lys/vinbigdata"

DATASET_NAME="padchest"              # 对应 stage1 里的 --dataname
DATASET_NAME_UPPER="PADCHEST-LT"
DATA_DIR="/data/padchest"
#NUM_CLASS=28

EXP_DIR="./experiment/new_stage1/5.8_0.95_4_0.6"  # 建议改个名字，

# ================= 2. 方法选择配置 =================
STAGE2_METHOD="splicemix-cl"             # 配置为 "splicemix" 或 "splicemix-cl" 或 baseline
NUM_CLASS=28                             # 注意：VinBigData 通常是 14 种疾病 + 1 个正常类，确认你的网络定义支持 15

# ================= 3. 动态生成输出目录 =================
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}/stage1_${STAGE2_METHOD}"

mkdir -p "${STAGE1_OUT}"

echo "==================================================="
echo "  [STEP 1] 启动 Stage 1: DINOv2 表征学习与末期多源证据去噪"
echo "  ==> 数据集: ${DATASET_NAME_UPPER}"
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="

# 执行 Python 脚本
python stage1_main_dinov2.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  -cd ${GPU_ID} \
  --img_size 224 \
  --num_class ${NUM_CLASS} \
  --workers 8 \
  --pretrained \
  --optim AdamW \
  --warm_up_epochs 6 \
  --fkl_consecutive_epochs 5