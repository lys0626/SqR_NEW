GPU_ID=2
# export CUDA_VISIBLE_DEVICES=${GPU_ID}      # 指定使用的单卡 GPU 编号

# DATASET_NAME="mimic"                     # 数据集名称小写 (给 Stage1 用: mimic, nih 等)
# DATASET_NAME_UPPER="MIMIC"               # 数据集名称大写 (给 Stage2 用: MIMIC, NIH-CHEST)
# DATA_DIR="/data/mimic_cxr/PA/7_1_2"      # 数据集的根目录路径

# DATASET_NAME="nih"
# DATASET_NAME_UPPER="NIH-CHEST"
# DATA_DIR="/data/nih-chest-xrays"

DATASET_NAME="chexpert"
DATASET_NAME_UPPER="CHEXPERT"
DATA_DIR="/data/chexpert_224"


EXP_DIR="/data/dsj/lys/SqR-NEW/experiment/CHEXPERT_13/"        # 实验输出的顶层根目录

# ================= 2. 方法选择配置 =================
# 可选值: "splicemix" 或 "splicemix-cl"，或 baseline
STAGE2_METHOD="splicemix-cl"             
STAGE2_OUT="${EXP_DIR}/${DATASET_NAME}/stage2_${STAGE2_METHOD}"
mkdir -p "${STAGE2_OUT}"
#指定软标签的路径
SOFT_LABEL_PATH=""
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
# --use_stage1_reliability
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
  -cd ${GPU_ID} \
  --soft_label_path "${SOFT_LABEL_PATH}" \
 

echo "==================================================="
echo "  训练流水线全部完成！Best Model 保存在 ${STAGE2_OUT} 中。"
echo "==================================================="