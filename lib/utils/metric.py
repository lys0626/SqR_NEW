import numpy as np
import torch
from sklearn import metrics

def check_tensor(tensor):
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    return tensor

class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename=None):
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

        if filename is not None:
            if isinstance(filename, list):
                self.filenames.extend(filename)
            else:
                self.filenames.append(filename)

    def compute_all_metrics(self):
        """
        计算 ws-MulSupCon 风格的指标: mAUC, Micro/Macro P/R/F1
        同时新增了 Label-wise Binary Accuracy (mAcc) 及其 per-class 列表
        """
        if self.scores.numel() == 0:
            return {}

        y_scores = self.scores.numpy()
        y_true = self.targets.numpy()
        y_true[y_true == -1] = 0  # 确保没有 -1 标签

        # 1. 计算 mAUC (及 per-class AUC)
        y_probs = 1 / (1 + np.exp(-y_scores))  # Sigmoid
        
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

        # 2. 计算预测标签 (阈值设为 0.5)
        y_pred = (y_probs >= 0.2).astype(int)

        # ========================================================
        # 3. 新增: 计算 Label-wise Binary Accuracy (方式 B)
        # ========================================================
        acc_list = []
        for i in range(y_true.shape[1]):
            # 独立计算每个类别的二元分类准确率 (TP + TN) / Total
            acc = metrics.accuracy_score(y_true[:, i], y_pred[:, i])
            acc_list.append(acc)
            
        mAcc = np.mean(acc_list) # 计算平均 Label-wise Accuracy

        # 4. 计算 P, R, F1 (Micro / Macro)
        return {
            'mAUC': mAUC,
            'auc_list': auc_list,
            'mAcc': mAcc,             # <--- 新增: 平均 Label-wise 二元准确率
            'acc_list': acc_list,     # <--- 新增: 每类二元准确率列表
            'micro_P': metrics.precision_score(y_true, y_pred, average='micro', zero_division=0),
            'micro_R': metrics.recall_score(y_true, y_pred, average='micro', zero_division=0),
            'micro_F1': metrics.f1_score(y_true, y_pred, average='micro', zero_division=0),
            'macro_P': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_R': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_F1': metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        }