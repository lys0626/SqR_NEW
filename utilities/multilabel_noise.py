import numpy as np
# 按照比例进行假阳性和假阴性的翻转
def inject_multilabel_noise(clean_labels, noise_type='asym', sym_rate=0.2, fn_rate=0.3, fp_rate=0.05, random_seed=42, no_finding_idx=14):
    """
    终极版多标签噪声注入器 (支持一键切换对称/非对称，且保证临床逻辑互斥)
    """
    np.random.seed(random_seed)
    noisy_labels = clean_labels.copy()
    
    # === 关键步骤 1：我们只对“疾病标签”（前14个类别）进行随机干扰 ===
    disease_indices = list(range(clean_labels.shape[1]))
    if no_finding_idx is not None:
        disease_indices.remove(no_finding_idx)
        
    disease_clean = clean_labels[:, disease_indices]
    disease_noisy = disease_clean.copy()
    
    if noise_type == 'sym':
        flip_mask = np.random.rand(*disease_clean.shape) < sym_rate
        disease_noisy = np.logical_xor(disease_noisy, flip_mask).astype(np.float32)
        
    elif noise_type == 'asym':
        # 针对疾病的漏诊 (1 -> 0)
        ones_mask = (disease_clean == 1)
        fn_flip = (np.random.rand(*disease_clean.shape) < fn_rate) & ones_mask
        
        # 针对疾病的误诊 (0 -> 1)
        zeros_mask = (disease_clean == 0)
        fp_flip = (np.random.rand(*disease_clean.shape) < fp_rate) & zeros_mask
        
        disease_noisy[fn_flip] = 0.0
        disease_noisy[fp_flip] = 1.0
    else:
        raise ValueError("noise_type 必须是 'sym' 或 'asym'")
        
    # 把污染后的疾病标签放回总矩阵
    noisy_labels[:, disease_indices] = disease_noisy
    
    # === 关键步骤 2：强行修正 'No finding' 逻辑 ===
    if no_finding_idx is not None:
        # 如果某张图的 14 个疾病标签加起来等于 0，那它就是健康的，No finding 必须强行设为 1
        # 如果 14 个疾病标签加起来大于 0，说明它有病了，No finding 必须强行设为 0
        disease_sums = np.sum(noisy_labels[:, disease_indices], axis=1)
        noisy_labels[:, no_finding_idx] = (disease_sums == 0).astype(np.float32)
        
    # 计算实际的总污染率 (此时涵盖了所有 15 个类别被翻转的数量)
    actual_total_noise_rate = np.mean(noisy_labels != clean_labels)
    
    print(f"[Noise Injector] 模式: {noise_type.upper()}")
    print(f"[Noise Injector] 临床逻辑修复: 'No finding' 类已根据疾病状态自动校正！")
    print(f"[Noise Injector] 实际总体标签破坏率: {actual_total_noise_rate:.4f}")
    
    return noisy_labels, actual_total_noise_rate, noise_type

