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
    #对-1标签默认为不参与判断
    def compute_all_metrics(self):
        """
        计算 ws-MulSupCon 风格的指标: mAUC, Micro/Macro P/R/F1
        采用 U-Ignore 策略：计算指标时严格剔除真实标签为 -1 的样本
        """
        if self.scores.numel() == 0:
            return {}

        y_scores = self.scores.numpy()
        y_true = self.targets.numpy()
        
        # 【修改核心】：删除 y_true[y_true == -1] = 0 这行代码，保留真实状态

        # 1. 计算预测概率与二值预测
        y_probs = 1 / (1 + np.exp(-y_scores)) # Sigmoid
        y_pred = (y_probs >= 0.5).astype(int)
        
        auc_list = []
        class_acc_list = []
        macro_P_list, macro_R_list, macro_F1_list = [], [], []
        
        # ====== 针对每个疾病类别单独应用 U-Ignore 掩码 ======
        for i in range(y_true.shape[1]):
            val_auc = -1.0
            
            # 动态掩码：找出当前类别中不是 -1 的有效样本
            valid_mask = (y_true[:, i] != -1)
            
            valid_y_true = y_true[valid_mask, i]
            valid_y_probs = y_probs[valid_mask, i]
            valid_y_pred = y_pred[valid_mask, i]
            
            # --- 1. 计算 Per-Class AUC ---
            try:
                # 必须同时存在明确的正类和负类才能计算 AUC
                if len(np.unique(valid_y_true)) == 2:
                    val_auc = metrics.roc_auc_score(valid_y_true, valid_y_probs)
            except ValueError:
                pass
            auc_list.append(val_auc)
            
            # --- 2. 计算 Per-Class ACC & P/R/F1 ---
            if len(valid_y_true) > 0:
                class_acc_list.append(np.mean(valid_y_pred == valid_y_true))
                macro_P_list.append(metrics.precision_score(valid_y_true, valid_y_pred, average='binary', zero_division=0))
                macro_R_list.append(metrics.recall_score(valid_y_true, valid_y_pred, average='binary', zero_division=0))
                macro_F1_list.append(metrics.f1_score(valid_y_true, valid_y_pred, average='binary', zero_division=0))
            else:
                class_acc_list.append(0.0)
                macro_P_list.append(0.0)
                macro_R_list.append(0.0)
                macro_F1_list.append(0.0)

        # 汇总宏平均指标 (Macro & mAUC)
        valid_aucs = [a for a in auc_list if a != -1.0]
        mAUC = np.mean(valid_aucs) if valid_aucs else 0.0
        mean_acc = np.mean(class_acc_list)

        # --- 3. 计算 Micro 指标 ---
        # 展平矩阵，并使用全局掩码剔除所有 -1 元素
        global_valid_mask = (y_true != -1)
        global_y_true = y_true[global_valid_mask]
        global_y_pred = y_pred[global_valid_mask]
        
        micro_P = metrics.precision_score(global_y_true, global_y_pred, average='binary', zero_division=0)
        micro_R = metrics.recall_score(global_y_true, global_y_pred, average='binary', zero_division=0)
        micro_F1 = metrics.f1_score(global_y_true, global_y_pred, average='binary', zero_division=0)

        return {
            'mAUC': mAUC,
            'auc_list': auc_list,
            'mean_ACC': mean_acc,                
            'class_ACC': class_acc_list,     
            'micro_P': micro_P,
            'micro_R': micro_R,
            'micro_F1': micro_F1,
            'macro_P': np.mean(macro_P_list),
            'macro_R': np.mean(macro_R_list),
            'macro_F1': np.mean(macro_F1_list)
        }
    

    #对-1标签认为是0,参与测试集和验证集判定
    # def compute_all_metrics(self):
    #     """
    #     计算 ws-MulSupCon 风格的指标: mAUC, Micro/Macro P/R/F1
    #     同时返回 auc_list 以供打印每类指标
    #     """
    #     if self.scores.numel() == 0:
    #         return {}

    #     y_scores = self.scores.numpy()
    #     y_true = self.targets.numpy()
    #     y_true[y_true == -1] = 0 # 确保没有 -1 标签

    #     # 1. 计算 mAUC (及 per-class AUC)
    #     y_probs = 1 / (1 + np.exp(-y_scores)) # Sigmoid
        
    #     auc_list = []
    #     for i in range(y_true.shape[1]):
    #         val = -1.0
    #         try:
    #             # 必须同时存在正类和负类才能计算 AUC
    #             if len(np.unique(y_true[:, i])) == 2:
    #                 val = metrics.roc_auc_score(y_true[:, i], y_probs[:, i])
    #         except ValueError:
    #             pass
    #         auc_list.append(val)
        
    #     # 计算平均值时只考虑有效的 AUC
    #     valid_aucs = [a for a in auc_list if a != -1.0]
    #     mAUC = np.mean(valid_aucs) if valid_aucs else 0.0

    #     # 2. 计算 P, R, F1 (Micro / Macro)
    #     # 阈值 0.5
    #     y_pred = (y_probs >= 0.5).astype(int)
    #     # ================= 改正的 ACC 计算 =================
    #     # 对每个类别单独计算 ACC（标签维度）
    #     class_acc = np.array([
    #         np.mean(y_pred[:, i] == y_true[:, i]) 
    #         for i in range(y_true.shape[1])
    #     ])
    #     mean_acc = np.mean(class_acc)
    #     # ====================================================
    #     return {
    #         'mAUC': mAUC,
    #         'auc_list': auc_list, # <--- 新增: 返回每类 AUC 列表
    #         'mean_ACC': mean_acc,                # <--- 新增: 返回 mean_ACC
    #         'class_ACC': class_acc.tolist(),     # <--- 新增: 返回每类 ACC 列表
    #         'micro_P': metrics.precision_score(y_true, y_pred, average='micro', zero_division=0),
    #         'micro_R': metrics.recall_score(y_true, y_pred, average='micro', zero_division=0),
    #         'micro_F1': metrics.f1_score(y_true, y_pred, average='micro', zero_division=0),
    #         'macro_P': metrics.precision_score(y_true, y_pred, average='macro', zero_division=0),
    #         'macro_R': metrics.recall_score(y_true, y_pred, average='macro', zero_division=0),
    #         'macro_F1': metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
    #     }

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