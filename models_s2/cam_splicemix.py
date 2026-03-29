import torch
import torch.nn.functional as F

class CAMSpliceMixer_MixedLabel:
    """
    【版本3：混合模式】精细的标签融合策略
    
    核心思想：
    1. 【图像】从借用样本复制该象限的图像块
    2. 【标签】只融合两个样本的"共同病灶"标签
       - 若两个样本都有某病灶 → 保留该病灶标签
       - 若只有一个样本有某病灶 → 可选地添加该标签
    
    这样做的优势：
    - ✅ 模型看到多个来源的相同病灶特征 → 更鲁棒
    - ✅ 避免引入不相关的病灶标签 → 减少噪声污染
    - ✅ 共同病灶优先级高，新病灶低优先级 → 精细控制
    """
    
    def __init__(self, use_new_labels=True, new_label_weight=1.0):
        """
        Args:
            use_new_labels: 是否添加借用样本的新病灶标签
                True: 融合所有病灶（包括新病灶）← 更鲁棒
                False: 只保留共同病灶 ← 更保守
            new_label_weight: 新病灶的融合权重 (0-1)
                1.0: 新病灶完全融合 (max 操作)
                0.5: 新病灶以概率融合
        """
        self.use_new_labels = use_new_labels
        self.new_label_weight = new_label_weight
    
    def __call__(self, noisy_imgs, noisy_tgts, noisy_masks, clean_imgs=None, clean_tgts=None, clean_masks=None, verbose=False):
        """
        Args:
            noisy_imgs: (B_n, C, H, W)
            noisy_tgts: (B_n, num_classes) - Stage 1 校正后的干净标签
            noisy_masks: (B_n, 2, 2) - True: 干净, False: 需要裁掉
            clean_imgs, clean_tgts, clean_masks: 干净样本池（可选）
            verbose: 是否打印调试信息
        
        Returns:
            X_syn: (B_n, C, H, W) - 拼接后的图像
            Y_syn: (B_n, num_classes) - 混合融合的标签
            source_grid: (B_n, 2, 2) - 溯源信息
        """
        B_n, C, H, W = noisy_imgs.shape
        half_h, half_w = H // 2, W // 2
        device = noisy_imgs.device
        
        # 初始化
        X_syn = noisy_imgs.clone()
        Y_syn = noisy_tgts.clone()
        
        source_grid = torch.arange(B_n, device=device).view(B_n, 1, 1).expand(B_n, 2, 2).clone()
        
        # 数据类型兼容性处理
        if noisy_masks.dtype != torch.bool:
            noisy_masks = noisy_masks.bool()
        if clean_masks is not None and clean_masks.dtype != torch.bool:
            clean_masks = clean_masks.bool()
        
        # 象限定义
        quadrants = [
            (0, 0, slice(0, half_h), slice(0, half_w)),   # TL
            (0, 1, slice(0, half_h), slice(half_w, W)),   # TR
            (1, 0, slice(half_h, H), slice(0, half_w)),   # BL
            (1, 1, slice(half_h, H), slice(half_w, W))    # BR
        ]
        
        # 统计信息（用于 verbose）
        stats = {
            'total_replaced': 0,
            'from_noisy': 0,
            'from_clean': 0,
            'from_black': 0,
            'labels_updated': 0
        }

        # ========================================================
        # 【核心处理】逐象限进行混合融合
        # ========================================================
        for r, c, slice_h, slice_w in quadrants:
            # 识别需要替换的象限
            needs_replace_mask = ~noisy_masks[:, r, c]
            needs_replace_idx = torch.where(needs_replace_mask)[0]
            
            # 识别可用的资源
            noisy_candidates = torch.where(noisy_masks[:, r, c])[0]
            
            clean_candidates = torch.empty(0, dtype=torch.long, device=device)
            if clean_imgs is not None and clean_masks is not None:
                clean_candidates = torch.where(clean_masks[:, r, c])[0]

            # ====================================================
            # 处理每个需要替换的象限
            # ====================================================
            for idx in needs_replace_idx:
                # 【优先级1】从其他噪声样本借用
                valid_noisy_cands = noisy_candidates[noisy_candidates != idx]
                
                if len(valid_noisy_cands) > 0:
                    # 随机选择一个来源样本
                    chosen_idx = torch.randint(0, len(valid_noisy_cands), (1,), device=device).item()
                    chosen = valid_noisy_cands[chosen_idx].item()
                    
                    # ✨ 【图像替换】
                    X_syn[idx, :, slice_h, slice_w] = noisy_imgs[chosen, :, slice_h, slice_w]
                    
                    # ✨ 【标签混合融合】
                    Y_syn[idx] = self._merge_labels(
                        Y_syn[idx],              # 自己的干净标签
                        noisy_tgts[chosen],      # 借用样本的干净标签
                        idx                      # 用于随机性
                    )
                    
                    source_grid[idx, r, c] = chosen
                    stats['total_replaced'] += 1
                    stats['from_noisy'] += 1
                
                # 【优先级2】从干净样本池借用
                elif len(clean_candidates) > 0:
                    chosen_idx = torch.randint(0, len(clean_candidates), (1,), device=device).item()
                    chosen = clean_candidates[chosen_idx].item()
                    
                    # ✨ 【图像替换】
                    X_syn[idx, :, slice_h, slice_w] = clean_imgs[chosen, :, slice_h, slice_w]
                    
                    # ✨ 【标签混合融合】
                    Y_syn[idx] = self._merge_labels(
                        Y_syn[idx],
                        clean_tgts[chosen],
                        idx
                    )
                    
                    source_grid[idx, r, c] = -1  # 标记来自干净样本池
                    stats['total_replaced'] += 1
                    stats['from_clean'] += 1
                
                # 【优先级3】黑色背景
                else:
                    X_syn[idx, :, slice_h, slice_w] = 0.0
                    # 标签保持不变（不添加新标签）
                    source_grid[idx, r, c] = -2
                    stats['total_replaced'] += 1
                    stats['from_black'] += 1
        
        if verbose:
            print(f"\n[CAMSpliceMixer V3 - 混合标签模式]")
            print(f"  总共替换象限数: {stats['total_replaced']}")
            print(f"    - 从噪声样本: {stats['from_noisy']}")
            print(f"    - 从干净样本: {stats['from_clean']}")
            print(f"    - 黑色背景: {stats['from_black']}")
        
        return X_syn, Y_syn, source_grid
    
    def _merge_labels(self, own_label, source_label, seed):
        """
        【核心】混合标签融合策略
        
        Args:
            own_label: [num_classes] - 自己的干净标签
            source_label: [num_classes] - 借用样本的干净标签
            seed: 用于随机性的种子
        
        Returns:
            merged_label: [num_classes] - 融合后的标签
        """
        # ========================================================
        # 【第一步】识别标签类型
        # ========================================================
        
        # 共同标签：两个样本都有的病灶
        common_labels = own_label & source_label  # [num_classes]
        
        # 自己独有的标签
        own_only_labels = own_label & ~source_label
        
        # 借用样本独有的标签（新病灶）
        new_labels = source_label & ~own_label
        
        # ========================================================
        # 【第二步】决定是否融合新病灶
        # ========================================================
        
        if self.use_new_labels:
            # 【鲁棒性优先】融合所有病灶
            if self.new_label_weight == 1.0:
                # 完全融合：使用 max（所有病灶）
                merged_label = torch.max(own_label, source_label)
            else:
                # 条件融合：新病灶以概率融合
                # new_labels 中，有些会被保留，有些不会
                prob = torch.full_like(new_labels, self.new_label_weight, dtype=torch.float32)
                random_mask = torch.bernoulli(prob).bool()
                
                # 融合：共同标签 + 自己的标签 + (条件)新标签
                merged_label = own_label.float()
                merged_label = merged_label | (common_labels.float())
                merged_label = merged_label | (new_labels.float() * random_mask.float())
                merged_label = merged_label.bool()
        else:
            # 【保守优先】只保留共同标签和自己的标签
            # merged_label = own_label | common_labels
            # 实际上就是 own_label（因为 common_labels ⊆ own_label）
            merged_label = own_label
        
        # ========================================================
        # 【第三步】返回融合后的标签
        # ========================================================
        return merged_label.float()


# =====================================================================
# 【加强版】支持更细粒度控制的版本
# =====================================================================

class CAMSpliceMixer_MixedLabel_Advanced(CAMSpliceMixer_MixedLabel):
    """
    【加强版V3】支持象限-类别级别的细粒度融合
    
    额外功能：
    - 可以针对不同的类别设置不同的融合策略
    - 支持基于 CAM 权重的智能选择
    """
    
    def __init__(self, use_new_labels=True, new_label_weight=1.0, 
                 class_specific_weights=None):
        """
        Args:
            class_specific_weights: dict, 针对特定类别的融合权重
                例如: {0: 0.9, 1: 0.5}  表示类别0权重0.9，类别1权重0.5
        """
        super().__init__(use_new_labels, new_label_weight)
        self.class_specific_weights = class_specific_weights or {}
    
    def _merge_labels_per_class(self, own_label, source_label, seed):
        """
        按类别精细融合
        """
        num_classes = len(own_label)
        merged_label = own_label.clone().float()
        
        for c in range(num_classes):
            # 获取该类别的融合权重
            weight = self.class_specific_weights.get(c, self.new_label_weight)
            
            # 检查是否是新增标签
            is_new = (source_label[c] == 1) and (own_label[c] == 0)
            
            if is_new:
                # 以 weight 的概率融合该新标签
                if torch.rand(1).item() < weight:
                    merged_label[c] = 1.0
        
        return merged_label

# import torch
# import torch.nn.functional as F

# class CAMSpliceMixer:
#     def __init__(self):
#         pass

#     def __call__(self, images, targets, cam_masks):
#         """
#         处理纯噪声 Batch 的 2x2 固定位置物理拼接
#         images: (B, C, H, W)
#         targets: (B, num_classes)
#         cam_masks: (B, 2, 2) 布尔型，指示四个象限是否有病灶
#         """
#         B, C, H, W = images.shape
#         half_h, half_w = H // 2, W // 2
#         device = images.device

#         # 1. 利用 torch.roll 形成 4 个备选池
#         img_pools = [torch.roll(images, shifts=i, dims=0) for i in range(4)]
#         tgt_pools = [torch.roll(targets, shifts=i, dims=0) for i in range(4)]
#         mask_pools = [torch.roll(cam_masks, shifts=i, dims=0) for i in range(4)]

#         X_syn = torch.zeros_like(images)
#         Y_syn = torch.zeros_like(targets)
#         source_grid = torch.zeros((B, 2, 2), dtype=torch.long, device=device) # 溯源网格记录 index 偏移

#         # 2. 定义 4 个格子的切片坐标
#         quadrants = [
#             (0, 0, slice(0, half_h), slice(0, half_w)),   # TL
#             (0, 1, slice(0, half_h), slice(half_w, W)),   # TR
#             (1, 0, slice(half_h, H), slice(0, half_w)),   # BL
#             (1, 1, slice(half_h, H), slice(half_w, W))    # BR
#         ]

#         # 3. 动态网格分配
#         for q_idx, (r, c, slice_h, slice_w) in enumerate(quadrants):
#             filled = torch.zeros(B, dtype=torch.bool, device=device)
            
#             # 优先用有病灶的图填补
#             for pool_idx in range(4):
#                 unfilled_mask = ~filled
#                 # 当前 pool 在当前格子 (r, c) 有病灶，且该 batch 样本该格子还没被填
#                 has_lesion = mask_pools[pool_idx][:, r, c]
#                 valid_to_fill = unfilled_mask & has_lesion
                
#                 if valid_to_fill.any():
#                     X_syn[valid_to_fill, :, slice_h, slice_w] = img_pools[pool_idx][valid_to_fill, :, slice_h, slice_w]
#                     Y_syn[valid_to_fill] = torch.max(Y_syn[valid_to_fill], tgt_pools[pool_idx][valid_to_fill])
#                     source_grid[valid_to_fill, r, c] = pool_idx # 记录来源于哪个 pool
#                     filled[valid_to_fill] = True
            
#             # 如果 4 张图都没有病灶，用 pool 0 的图补天（无病灶背景）
#             needs_bg = ~filled
#             if needs_bg.any():
#                 X_syn[needs_bg, :, slice_h, slice_w] = img_pools[0][needs_bg, :, slice_h, slice_w]
#                 # 背景没有病灶，不更新 Y_syn
#                 source_grid[needs_bg, r, c] = 0

#         return X_syn, Y_syn, source_grid