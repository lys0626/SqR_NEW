#!/bin/bash
set -e  # 只要发生任何报错，脚本立刻停止

# ================= 1. 基础全局配置 =================
GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}      

DATASET_NAME="nih"
DATASET_NAME_UPPER="NIH-CHEST"
DATA_DIR="/data/nih-chest-xrays"

EXP_DIR="./experiment/new"        
STAGE2_METHOD="splicemix-cl"             
NUM_CLASS=14

# ================= 2. 动态生成输出目录 =================
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}/stage1_${STAGE2_METHOD}_q2l"
# 确保目录存在 (虽然 Stage1 跑完应该已经存在了)
mkdir -p "${STAGE1_OUT}"

# ================= 3. 执行 CAM 掩码生成 =================
echo "==================================================="
echo "  [STEP 1.5] 启动离线 CAM 2x2 掩码生成"
echo "  ==> 读取/输出目录: ${STAGE1_OUT}"
echo "  ==> 使用显卡 ID: ${GPU_ID}"
echo "==================================================="

# 注意：必须要传入和 Stage 1 完全一致的模型结构参数
python generate_cam_masks.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --img_size 448 \
  --num_class ${NUM_CLASS} \
  --backbone resnet50 \
  --keep_input_proj \
  --hidden_dim 512 \
  --dim_feedforward 2048 \
  --dec_layers 2 \
  --enc_layers 1 \
  --nheads 4 \
  --batch-size 128 \
  --workers 16 

echo "==================================================="
echo "  CAM 掩码生成完毕！结果已保存在: ${STAGE1_OUT}/noise_cam_masks.pt"
echo "==================================================="