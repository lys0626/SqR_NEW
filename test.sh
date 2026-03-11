#!/bin/bash
# ==============================================================================
#  Unified Robust MLC - 单卡测试脚本 (动态寻址版)
# ==============================================================================

# ================= 1. 配置必须与训练时一致 =================
GPU_ID=1
DATASET_NAME="nih"
DATASET_NAME_UPPER="NIH-CHEST"
DATA_DIR="/data/nih-chest-xrays"
EXP_DIR="./experiment/robust_run"

STAGE2_METHOD="splicemix-cl"
# DATANAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"
# 配置为 "splicemix" 或 "splicemix-cl"  

# ================= 2. 动态拼接权重路径 =================
STAGE2_OUT="${EXP_DIR}/${DATASET_NAME}_stage2_${STAGE2_METHOD}"

# 推导模型名称参数与目录名称
if [ "$STAGE2_METHOD" = "splicemix-cl" ]; then
    # 完整版：数据增强 + 对比学习损失
    MODEL_ARG="SpliceMix_CL"
    MIXER_ARG="SpliceMix--Default=True"
    MODEL_DIR="SpliceMix_CL"
    METHOD_SUFFIX="splicemix"       # <--- 新增
    
elif [ "$STAGE2_METHOD" = "splicemix" ]; then
    # 消融版 1：仅使用 SpliceMix 数据增强，无对比损失
    MODEL_ARG="ResNet-50"
    MIXER_ARG="SpliceMix--Default=True"
    MODEL_DIR="ResNet_50"  # 注意这里的下划线，用于匹配内部生成的文件夹
    METHOD_SUFFIX="splicemix"       # <--- 新增
    
else
    # 纯净版 Baseline：无任何增强，无对比损失
    MODEL_ARG="ResNet-50"
    MIXER_ARG=""           # 置空，不触发 Mixer
    MODEL_DIR="ResNet_50"
    METHOD_SUFFIX="baseline"        # <--- 新增 (对应 engine.py) 
fi

# 根据 utils.py 的保存逻辑，实际目录为: <STAGE2_OUT>/<DATASET>/<MODEL_ARG>/
# 注意: engine.py 中保存的文件名写死为了 ${data_set}_splicemix_best.pt 
# (不管加没加 CL，文件名都是这个，依靠外层的模型文件夹区分)
# WEIGHTS_PATH="${STAGE2_OUT}/${DATASET_NAME_UPPER}/${MODEL_DIR}/${DATASET_NAME_UPPER}_${METHOD_SUFFIX}_best.pt"
WEIGHTS_PATH="/data/dsj/lys/SqR-NEW/experiment/robust_run/nih_stage2_splicemix-cl/NIH-CHEST/SpliceMix_CL/NIH-CHEST_splicemix_best.pt"
echo "==================================================="
echo "  启动测试评估模式 (Evaluate Only) "
echo "  ==> 评估方法: ${STAGE2_METHOD}"
echo "  ==> 加载权重: ${WEIGHTS_PATH}"
echo "==================================================="

# -e 0 : 强制进入 Evaluate 模式，不进行训练
python stage2_main.py \
  --data-set ${DATASET_NAME_UPPER} \
  --data-root ${DATA_DIR} \
  --model ${MODEL_ARG} \
  -mixer ${MIXER_ARG} \
  --batch-size 32 \
  -e 0 \
  -r ${WEIGHTS_PATH} \
  -cd ${GPU_ID}