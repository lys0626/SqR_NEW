set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=1
# 建议导出全局可见的 GPU ID，以防某些脚本没有适配 -cd 参数
export CUDA_VISIBLE_DEVICES=${GPU_ID}      

# DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
# DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
# DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径

DATASET_NAME="nih"
DATASET_NAME_UPPER="NIH-CHEST"
DATA_DIR="/data/nih-chest-xrays"

EXP_DIR="./experiment/new_knn"        # 实验输出的顶层根目录
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

# =========================================================
# ================= 新增: STEP 1.5 掩码生成 =================
# =========================================================
echo "==================================================="
echo "  [STEP 1.5] 启动离线 CAM 2x2 掩码生成"
echo "  ==> 读取/输出目录: ${STAGE1_OUT}"
echo "==================================================="

# 注意：必须要传入和 Stage 1 完全一致的模型结构参数，否则无法正确加载权重！
python generate_cam_masks.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --img_size 224 \
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
echo "  [STEP 2] 启动 Stage 2: 纯净子集 CNN 训练"
echo "  ==> 采用方法: ${STAGE2_METHOD}"
echo "  ==> 输出目录: ${STAGE2_OUT}"
echo "==================================================="

# 提取 Stage 1 产出的各种 pt 文件路径
CLEAN_IDX_PATH="${STAGE1_OUT}/clean_indices.pt"
NOISY_IDX_PATH="${STAGE1_OUT}/noisy_indices.pt"
CAM_MASK_PATH="${STAGE1_OUT}/noise_cam_masks.pt"

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
# 新增了 --clean_idx_path, --noisy_idx_path, --cam_mask_path 等参数传递
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
echo "  训练流水线全部完成！Best Model 保存在 ${STAGE2_OUT} 中。"
echo "==================================================="