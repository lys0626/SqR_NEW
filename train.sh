#!/bin/bash
# [新增] 强行让系统优先使用 lys2 虚拟环境里的 C++ 库！
export LD_LIBRARY_PATH="/home/dsj/anaconda3/envs/lys2/lib:$LD_LIBRARY_PATH"
# ================= 1. 基础全局配置 =================
GPU_ID=0                                 # 指定使用的单卡 GPU 编号
# DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
# DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
# DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径
EXP_DIR="./experiment/robust_run"        # 实验输出的顶层根目录
DATASET_NAME="nih"
DATASET_NAME_UPPER="NIH-CHEST"
DATA_DIR="/data/nih-chest-xrays"
# ================= 2. 方法选择配置 =================
# 可选值: "splicemix" 或 "splicemix-cl"
STAGE2_METHOD="splicemix-cl"             
NUM_CLASS=14
# ================= 3. 动态生成输出目录 =================
# 生成的目录样式示例: ./experiment/robust_run/mimic_stage1_q2l
# 生成的目录样式示例: ./experiment/robust_run/mimic_stage2_splicemix-cl
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}_stage1_${STAGE2_METHOD}_q2l"
STAGE2_OUT="${EXP_DIR}/${DATASET_NAME}_stage2_${STAGE2_METHOD}"

mkdir -p ${STAGE1_OUT}
mkdir -p ${STAGE2_OUT}

echo "==================================================="
echo "  [STEP 1] 启动 Stage 1: Q2L 预热与 RoLT 数据清洗    "
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="
python stage1_main.py \
  --dataname ${DATASET_NAME} \
  --dataset_dir ${DATA_DIR} \
  --output ${STAGE1_OUT} \
  --epochs 4 \
  --splicemix_start_epoch 2 \
  --rolt_start_epoch 1 \
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
  --scheduler StepLR


echo "==================================================="
echo "  [STEP 2] 启动 Stage 2: 纯净子集 CNN 训练"
echo "  ==> 采用方法: ${STAGE2_METHOD}"
echo "  ==> 输出目录: ${STAGE2_OUT}"
echo "==================================================="
CLEAN_IDX_PATH="${STAGE1_OUT}/clean_indices.pt"

# 推导模型名称参数与目录名称
if [ "$STAGE2_METHOD" = "splicemix-cl" ]; then
    # 完整版：数据增强 + 对比学习损失
    MODEL_ARG="SpliceMix_CL"
    MIXER_ARG="SpliceMix--Default=True"
    MODEL_DIR="SpliceMix_CL"
    
elif [ "$STAGE2_METHOD" = "splicemix" ]; then
    # 消融版 1：仅使用 SpliceMix 数据增强，无对比损失
    MODEL_ARG="ResNet-50"
    MIXER_ARG="SpliceMix--Default=True"
    MODEL_DIR="ResNet_50"  # 注意这里的下划线，用于匹配内部生成的文件夹
    
else
    # 纯净版 Baseline：无任何增强，无对比损失
    MODEL_ARG="ResNet-50"
    MIXER_ARG=""           # 置空，不触发 Mixer
    MODEL_DIR="ResNet_50"
fi

python stage2_main.py \
  --data-set ${DATASET_NAME_UPPER} \
  --data-root ${DATA_DIR} \
  --save-dir ${STAGE2_OUT} \
  --model ${MODEL_ARG} \
  -mixer ${MIXER_ARG} \
  --epochs 100 \
  --batch-size 32 \
  --optimizer SGD \
  --lr 0.05 \
  --warmup-epochs 5 \
  --clean_mask_path ${CLEAN_IDX_PATH} \
  -cd ${GPU_ID}

echo "==================================================="
echo "  训练流水线全部完成！Best Model 保存在 ${STAGE2_OUT} 中。"
echo "==================================================="