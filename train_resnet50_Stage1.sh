set -e  # <--- [强烈建议新增] 只要发生任何报错，脚本立刻停止，绝不往下瞎跑！

# ================= 1. 基础全局配置 =================
GPU_ID=5

DATASET_NAME="vinbigdata"                     # 对应 stage1 里的 args.dataname
DATASET_NAME_UPPER="VINBIGDATA"               
DATA_DIR="/data/dsj/lys/vinbigdata"
NUM_CLASS=15

EXP_DIR="./experiment/VINVIG_denoise/END/5_13_0.94_0.95_4_asym_0.2_0.2_0.85_0.2_all_loss_clean_0.55KNN"        # 实验输出的顶层根目录
STAGE2_METHOD="splicemix-cl"         

# ================= 3. 动态生成输出目录 =================
STAGE1_OUT="${EXP_DIR}/${DATASET_NAME}/stage1_${STAGE2_METHOD}"

mkdir -p "${STAGE1_OUT}"

echo "==================================================="
echo "  [STEP 1] 启动 Stage 1: Q2L 预热与 MEE(Early Cutting) 标签级清洗"
echo "  ==> 采用算法: ICCV2023 LateStopping + NeurIPS2025 MEE"
echo "  ==> 输出目录: ${STAGE1_OUT}"
echo "==================================================="
  # --noise_type asym \
  # --fn_rate 0.2 \
  # --fp_rate 0.2 \
  # --noise_type sym \
  # --sym_rate 0.2
  # --disable_fn_mining
python stage1_main_resnet50.py \
  --dataname "${DATASET_NAME}" \
  --dataset_dir "${DATA_DIR}" \
  --output "${STAGE1_OUT}" \
  --epochs 60 \
  --batch-size 128 \
  --lr 1e-4 \
  -cd ${GPU_ID} \
  --img_size 224 \
  --num_class ${NUM_CLASS} \
  --fkl_consecutive_epochs 8 \
  --optim AdamW \
  --early_cutting_rate 5 \
  --warm_up_epochs 0 \
  --inject_noise \
  --noise_type asym \
  --fn_rate 0.2 \
  --fp_rate 0.2 \

