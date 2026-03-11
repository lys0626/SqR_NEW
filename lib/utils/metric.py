# most borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py

import numpy as np
import torch
import numpy as np
from sklearn import metrics
def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(imagessetfilelist, num, return_each=False):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())
    
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:,num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims = True)


    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP

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