#!/bin/bash
# [新增] 强行让系统优先使用 lys2 虚拟环境里的 C++ 库！
# set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=5
# export CUDA_VISIBLE_DEVICES=${GPU_ID}      # 指定使用的单卡 GPU 编号

DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径

# DATASET_NAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"

EXP_DIR="./experiment/robust_run"        # 实验输出的顶层根目录

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
echo "  [STEP 1] 启动 Stage 1: Q2L 预热与 RoLT 数据清洗    "
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="

python stage1_main.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --epochs 15 \
  --splicemix_start_epoch 12 \
  --rolt_start_epoch 5 \
  --batch-size 32 \
  --optim AdamW \
  --lr 1e-4 \
  -cd ${GPU_ID} \
  --img_size 224 \
  --num_class ${NUM_CLASS} \
  --backbone resnet50 \
  --workers 4 \
  --pretrained \
  --momentum 0.9 \
  --keep_input_proj \
  --hidden_dim 512 \
  --dim_feedforward 2048 \
  --dec_layers 2 \
  --enc_layers 1 \
  --nheads 4 \
  --scheduler OneCycle

echo "==================================================="
echo "  [STEP 2] 启动 Stage 2: 纯净子集 CNN 训练"
echo "  ==> 采用方法: ${STAGE2_METHOD}"
echo "  ==> 输出目录: ${STAGE2_OUT}"
echo "==================================================="

CLEAN_IDX_PATH="${STAGE1_OUT}/clean_indices.pt"

# 推导模型名称参数与目录名称
if [ "$STAGE2_METHOD" = "splicemix-cl" ]; then
    MODEL_ARG="SpliceMix_CL"
    MIXER_ARG="SpliceMix--Default=True"
    MODEL_DIR="SpliceMix_CL"
elif [ "$STAGE2_METHOD" = "splicemix" ]; then
    MODEL_ARG="ResNet-50"
    MIXER_ARG="SpliceMix--Default=True"
    MODEL_DIR="ResNet_50"
else
    MODEL_ARG="ResNet-50"
    MIXER_ARG=""
    MODEL_DIR="ResNet_50"
fi

# 注意：这里的 -mixer 加上了双引号，防止参数为空时发生偏移报错
python stage2_main.py \
  --data-set "${DATASET_NAME_UPPER}" \
  --data-root "${DATA_DIR}" \
  --save-dir "${STAGE2_OUT}" \
  --model "${MODEL_ARG}" \
  -mixer "${MIXER_ARG}" \
  --epochs 100 \
  --batch-size 32 \
  --optimizer SGD \
  --lr 0.05 \
  --warmup-epochs 5 \
  --clean_mask_path "${CLEAN_IDX_PATH}" \
  -cd ${GPU_ID}

echo "==================================================="
echo "  训练流水线全部完成！Best Model 保存在 ${STAGE2_OUT} 中。"
echo "==================================================="