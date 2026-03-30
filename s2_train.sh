#!/bin/bash
set -e  # <--- 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=3
export CUDA_VISIBLE_DEVICES=${GPU_ID}      

# DATASET_NAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"
DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径
EXP_DIR="./experiment/loss_2"        
STAGE2_METHOD="splicemix-cl"             
NUM_CLASS=13

# ================= 2. 动态生成输出目录 =================
STAGE1_OUT="/data/dsj/lys/SqR-NEW/experiment/loss_2/mimic/stage1_splicemix-cl_q2l"
STAGE2_OUT="${EXP_DIR}/${DATASET_NAME}/stage2_${STAGE2_METHOD}"

mkdir -p "${STAGE2_OUT}"

# ================= 3. 提取交接文件路径并检查 =================
CLEAN_IDX_PATH="${STAGE1_OUT}/clean_indices.pt"
NOISY_IDX_PATH="${STAGE1_OUT}/noisy_indices.pt"
CAM_MASK_PATH="${STAGE1_OUT}/noise_cam_masks.pt"

echo "==================================================="
echo "  [前置检查] 验证 Stage 1 & 1.5 的产出文件..."
echo "==================================================="
for FILE in "$CLEAN_IDX_PATH" "$NOISY_IDX_PATH" "$CAM_MASK_PATH"; do
    if [ ! -f "$FILE" ]; then
        echo "❌ [致命错误] 找不到必须的文件: $FILE"
        echo "请确保你已经成功跑完了 Stage 1 和 generate_cam_masks.py！"
        exit 1
    fi
done
echo "✅ 所有交接文件检查通过！"

# ================= 4. 推导模型参数 =================
if [ "$STAGE2_METHOD" = "splicemix-cl" ]; then
    MODEL_ARG="SpliceMix_CL"
    MIXER_ARG="SpliceMix--Default=True"
elif [ "$STAGE2_METHOD" = "splicemix" ]; then
    MODEL_ARG="ResNet-50"
    MIXER_ARG="SpliceMix--Default=True"
else
    MODEL_ARG="ResNet-50"
    MIXER_ARG=""
fi

echo "==================================================="
echo "  [STEP 2] 启动 Stage 2: 纯净子集双轨训练"
echo "  ==> 采用方法: ${STAGE2_METHOD}"
echo "  ==> 输出目录: ${STAGE2_OUT}"
echo "==================================================="

# ================= 5. 启动训练 =================
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
  --clean_idx_path "${CLEAN_IDX_PATH}" \
  --noisy_idx_path "${NOISY_IDX_PATH}" \
  --cam_mask_path "${CAM_MASK_PATH}" \
  --clean_mask_path "${CLEAN_IDX_PATH}" \
  -cd ${GPU_ID}

echo "==================================================="
echo "  🎉 训练流水线全部完成！Best Model 保存在 ${STAGE2_OUT} 中。"
echo "==================================================="