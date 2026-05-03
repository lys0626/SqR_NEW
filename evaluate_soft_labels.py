import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score # 需要 pip install scikit-learn
def evaluate_stage1_correction():
    # 2. 加载三个张量：加上 weights_only=True 消除警告，加上 .cpu() 统一移动到内存
    try:
        clean_labels = torch.load(
            f'/data/dsj/lys/vinbigdata/clean_labels_gt.pt', 
            weights_only=True
        ).cpu()
        
        noisy_labels = torch.load(
            f'/data/dsj/lys/vinbigdata/noisy_labels_ASYM_FN0.2_FP0.2_Total0.230.pt', 
            weights_only=True
        ).cpu()
        
        soft_targets = torch.load(
            f'/data/dsj/lys/SqR-NEW/experiment/VINVIG/5_2_Resnet50_12_0.98_old_0.2_asym_resnet50/vinbigdata/stage1_splicemix-cl/asymmetric_soft_targets.pt', 
            weights_only=True
        ).cpu()
    except Exception as e:
        print(f"加载文件失败，请确保 Stage 1 已成功跑完且路径正确: {e}")
        return
    # === [新增] 真正意义上的软标签相似度评测 ===
    print("\n" + "="*40)
    print("🌟 软标签连续值相似度分析 (Soft-Level Metrics)")
    print("="*40)
    
    # 1. 计算 MSE (均方误差)
    # 将模型吃进去的极端 0/1 噪声和真实标签计算原始 MSE (做 Baseline)
    baseline_mse = F.mse_loss(noisy_labels.float(), clean_labels).item()
    # 计算模型生成的软标签和真实标签的 MSE
    soft_mse = F.mse_loss(soft_targets, clean_labels).item()
    
    print(f"原始噪声的数值误差 (MSE): {baseline_mse:.4f}")
    print(f"Stage 1 软标签数值误差 (MSE): {soft_mse:.4f}")
    if soft_mse < baseline_mse:
        print(f" -> 📉 MSE 降低了 {(baseline_mse - soft_mse) / baseline_mse:.2%}，数值分布更接近真实情况！")

    # 2. 计算全局 AUC-ROC (评估排序能力)
    # 为了防止某些类别在 batch 里全是 0 导致报错，我们将矩阵展平 (flatten) 计算全局 AUC
    try:
        y_true_flat = clean_labels.numpy().flatten()
        y_score_flat = soft_targets.numpy().flatten()
        
        auc_score = roc_auc_score(y_true_flat, y_score_flat)
        print(f"\n📊 软标签全局分辨力 (Global AUC): {auc_score:.4f}")
        if auc_score > 0.8:
            print(" -> 🏅 极佳！软标签对病灶和健康区域具有极强的分辨能力！")
        elif auc_score > 0.7:
            print(" -> 🥈 良好，足以指导 Stage 2 的训练。")
        else:
            print(" -> ⚠️ 一般，模型可能没有很好地学到特征。")
    except Exception as e:
        print(f"计算 AUC 失败: {e}")
    # 3. 将软标签进行二值化处理 (大于 0.5 视为预测有病灶)
    soft_preds = (soft_targets > 0.5).float()
    
    # 4. 计算全局噪声干扰率
    total_elements = clean_labels.numel()
    original_errors = (noisy_labels != clean_labels).sum().item()
    print(f"=== 实验数据概览 ===")
    print(f"总标签数量 (N x 15): {total_elements}")
    print(f"被噪声破坏的标签数: {original_errors} (污染率: {original_errors/total_elements:.2%})")
    
    # 5. 分析模型纠错能力
    corrected_mask = (soft_preds == clean_labels) & (noisy_labels != clean_labels)
    corrected_count = corrected_mask.sum().item()
    
    degraded_mask = (soft_preds != clean_labels) & (noisy_labels == clean_labels)
    degraded_count = degraded_mask.sum().item()
    
    print(f"\n=== Stage 1 (SqR-NEW) 软标签纠错分析 ===")
    print(f"✅ 成功从噪声中救回(纠正)的标签数: {corrected_count} (纠错恢复率: {corrected_count/original_errors:.2%})")
    print(f"❌ 原本干净但被模型改错的标签数: {degraded_count}")
    
    # 6. 计算最终的标签质量提升
    final_errors = (soft_preds != clean_labels).sum().item()
    print(f"\n📈 结论:")
    print(f"喂给模型的原始标签错误率: {original_errors/total_elements:.2%}")
    print(f"Stage 1 清洗后的标签错误率: {final_errors/total_elements:.2%}")
    if final_errors < original_errors:
        print("💡 恭喜！Stage 1 成功提升了数据集的标签质量，为 Stage 2 打下了更纯净的基础。")
    else:
        print("⚠️ 警告：清洗后的错误率反而上升了，可能需要调整 early_cutting_rate 或确认 LLM 矩阵设置。")

if __name__ == '__main__':
    evaluate_stage1_correction()