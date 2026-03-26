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
        preds_patches: (B, 4, num_classes)
        soft_targets: (B, 4, num_classes) - 来源于 4 个 pool 的 mother_soft_target
        source_grid: (B, 2, 2)
        """
        B = preds_patches.shape[0]
        grid_flat = source_grid.view(B, 4) 
        
        loss_cam = 0.0
        for b in range(B):
            for p_idx in range(4): 
                pool_idx = grid_flat[b, p_idx] 
                
                patch_pred = preds_patches[b, p_idx] 
                mother_soft_target = torch.sigmoid(soft_targets[b, pool_idx]) 
                
                loss_cam += F.binary_cross_entropy_with_logits(patch_pred, mother_soft_target)
                
        return loss_cam / (B * 4)