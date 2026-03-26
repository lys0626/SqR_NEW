import torch
import torch.nn.functional as F

class CAMSpliceMixer:
    def __init__(self):
        pass

    def __call__(self, images, targets, cam_masks):
        """
        处理纯噪声 Batch 的 2x2 固定位置物理拼接
        images: (B, C, H, W)
        targets: (B, num_classes)
        cam_masks: (B, 2, 2) 布尔型，指示四个象限是否有病灶
        """
        B, C, H, W = images.shape
        half_h, half_w = H // 2, W // 2
        device = images.device

        # 1. 利用 torch.roll 形成 4 个备选池
        img_pools = [torch.roll(images, shifts=i, dims=0) for i in range(4)]
        tgt_pools = [torch.roll(targets, shifts=i, dims=0) for i in range(4)]
        mask_pools = [torch.roll(cam_masks, shifts=i, dims=0) for i in range(4)]

        X_syn = torch.zeros_like(images)
        Y_syn = torch.zeros_like(targets)
        source_grid = torch.zeros((B, 2, 2), dtype=torch.long, device=device) # 溯源网格记录 index 偏移

        # 2. 定义 4 个格子的切片坐标
        quadrants = [
            (0, 0, slice(0, half_h), slice(0, half_w)),   # TL
            (0, 1, slice(0, half_h), slice(half_w, W)),   # TR
            (1, 0, slice(half_h, H), slice(0, half_w)),   # BL
            (1, 1, slice(half_h, H), slice(half_w, W))    # BR
        ]

        # 3. 动态网格分配
        for q_idx, (r, c, slice_h, slice_w) in enumerate(quadrants):
            filled = torch.zeros(B, dtype=torch.bool, device=device)
            
            # 优先用有病灶的图填补
            for pool_idx in range(4):
                unfilled_mask = ~filled
                # 当前 pool 在当前格子 (r, c) 有病灶，且该 batch 样本该格子还没被填
                has_lesion = mask_pools[pool_idx][:, r, c]
                valid_to_fill = unfilled_mask & has_lesion
                
                if valid_to_fill.any():
                    X_syn[valid_to_fill, :, slice_h, slice_w] = img_pools[pool_idx][valid_to_fill, :, slice_h, slice_w]
                    Y_syn[valid_to_fill] = torch.max(Y_syn[valid_to_fill], tgt_pools[pool_idx][valid_to_fill])
                    source_grid[valid_to_fill, r, c] = pool_idx # 记录来源于哪个 pool
                    filled[valid_to_fill] = True
            
            # 如果 4 张图都没有病灶，用 pool 0 的图补天（无病灶背景）
            needs_bg = ~filled
            if needs_bg.any():
                X_syn[needs_bg, :, slice_h, slice_w] = img_pools[0][needs_bg, :, slice_h, slice_w]
                # 背景没有病灶，不更新 Y_syn
                source_grid[needs_bg, r, c] = 0

        return X_syn, Y_syn, source_grid