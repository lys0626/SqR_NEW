import torch
import numpy as np
from sklearn import metrics

class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename):
        output = check_tensor(output)
        target = check_tensor(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        if target.dim() == 1:
            target = target.view(-1, 1)

        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = int((self.scores.storage().size() + output.numel()) * 1.5)
            self.scores.storage().resize_(new_size)
            self.targets.storage().resize_(new_size)

        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        self.filenames += filename

    def compute_all_metrics(self):
        """
        计算 ws-MulSupCon 风格的指标: mAUC, Micro/Macro P/R/F1
        同时返回 auc_list 以供打印每类指标
        """
        if self.scores.numel() == 0:
            return {}

        y_scores = self.scores.numpy()
        y_true = self.targets.numpy()
        y_true[y_true == -1] = 0 # 确保没有 -1 标签

        # 1. 计算 mAUC (及 per-class AUC)
        y_probs = 1 / (1 + np.exp(-y_scores)) # Sigmoid
        
        auc_list = []
        for i in range(y_true.shape[1]):
            val = -1.0
            try:
                # 必须同时存在正类和负类才能计算 AUC
                if len(np.unique(y_true[:, i])) == 2:
                    val = metrics.roc_auc_score(y_true[:, i], y_probs[:, i])
            except ValueError:
                pass
            auc_list.append(val)
        
        # 计算平均值时只考虑有效的 AUC
        valid_aucs = [a for a in auc_list if a != -1.0]
        mAUC = np.mean(valid_aucs) if valid_aucs else 0.0

        # 2. 计算 P, R, F1 (Micro / Macro)
        # 阈值 0.5
        y_pred = (y_probs >= 0.5).astype(int)

        return {
            'mAUC': mAUC,
            'auc_list': auc_list, # <--- 新增: 返回每类 AUC 列表
            'micro_P': metrics.precision_score(y_true, y_pred, average='micro', zero_division=0),
            'micro_R': metrics.recall_score(y_true, y_pred, average='micro', zero_division=0),
            'micro_F1': metrics.f1_score(y_true, y_pred, average='micro', zero_division=0),
            'macro_P': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_R': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_F1': metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

def check_tensor(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    return tensor

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
    def add(self, val):
        self.val = val
        self.sum += val
        self.count += 1
    def average(self):
        self.avg = self.sum / self.count
        return self.avg
    def value(self):
        return self.val