import torch.nn as nn
import torch
import torch.nn.functional as F
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

    def forward(self, input, target):
        # input: bs, nc; the output of model without sigmoid
        # target: bs, nc; multi-hot format

        return self.loss_fn(input, target)
class Loss_fn_CAM(nn.Module):
    def __init__(self):
        super(Loss_fn_CAM, self).__init__()

    def forward(self, preds_patches, soft_targets, source_grid):
        """
        全张量向量化加速版本！极速运算，无 Python for 循环。
        preds_patches: (B, 4, num_classes)
        soft_targets: (B, 4, num_classes)
        source_grid: (B, 2, 2)
        """
        B, num_patches, num_classes = preds_patches.shape
        
        # 1. 展平网格，找到对应的 pool_idx. shape: (B, 2, 2) -> (B, 4)
        grid_flat = source_grid.view(B, 4) 
        
        # 2. 将 grid_flat 扩展第三个维度，以便通过 gather 抽取所有的类别通道
        # grid_expanded shape: (B, 4, num_classes)
        grid_expanded = grid_flat.unsqueeze(-1).expand(B, 4, num_classes)
        
        # 3. 瞬间抽取整个 Batch 的 mother_soft_target
        selected_soft_targets = torch.gather(soft_targets, dim=1, index=grid_expanded)
        mother_soft_targets = torch.sigmoid(selected_soft_targets)
        
        # 4. 一次性计算所有的 BCE Loss 并求均值！
        loss_cam = F.binary_cross_entropy_with_logits(preds_patches, mother_soft_targets)
                
        return loss_cam