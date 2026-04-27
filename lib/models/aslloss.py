"""
    Most borrow from: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps, max=1-self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricLossOptimized(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        targets = y
        anti_targets = 1 - y
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss.add_(anti_targets * torch.log(xs_neg.clamp(min=self.eps)))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    xs_pos = xs_pos * targets
                    xs_neg = xs_neg * anti_targets
                    asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                            self.gamma_pos * targets + self.gamma_neg * anti_targets)
                loss *= asymmetric_w
            else:
                xs_pos = xs_pos * targets
                xs_neg = xs_neg * anti_targets
                asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                            self.gamma_pos * targets + self.gamma_neg * anti_targets)   
                loss *= asymmetric_w         
                
        _loss = - loss.sum() / x.size(0)
        _loss = _loss / y.size(1) 

        return _loss
class DynamicAsymmetricLoss(nn.Module):
    # 【修改点 1】：将 eps=1e-8 改为 eps=1e-5，完美兼容 AMP float16
    def __init__(self, gamma_neg=4, gamma_pos=1, base_clip=0.05, max_clip=0.2, eps=1e-5):
        super(DynamicAsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.base_clip = base_clip
        self.max_clip = max_clip
        self.eps = eps

    def forward(self, x, y, progress_ratio=0.0):
        """
        progress_ratio: 当前训练进度 (current_epoch / total_epochs)，取值 [0, 1]
        """
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        # 动态 Asymmetric Clipping: 随训练深入，逐步提升 clip 阈值，拒绝拟合后期暴露的顽固噪声
        dynamic_clip = self.base_clip + progress_ratio * (self.max_clip - self.base_clip)
        if dynamic_clip > 0:
            xs_neg = (xs_neg + dynamic_clip).clamp(max=1)

        # 【修改点 2】：移除 max=1-self.eps，只保留 min=self.eps 防护下溢出 (防 log(0) 变 -inf)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # 动态 Gamma 聚焦: 训练后期加大对多数类（易学负样本）的抑制，释放长尾正样本的梯度空间
        dynamic_gamma_neg = self.gamma_neg * (1.0 + progress_ratio)
        
        with torch.no_grad():
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + dynamic_gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        loss = (los_pos + los_neg) * one_sided_w
        return -loss.sum() / x.size(0)