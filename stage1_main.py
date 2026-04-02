import argparse
import os, sys
import random
import datetime
import time
import json
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

from lib.dataset.get_dataset import get_datasets
from lib.utils.logger import setup_logger
from lib.models.query2label import build_q2l
from lib.utils.metric import AveragePrecisionMeter
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict

from lib.models.aslloss import AsymmetricLossOptimized

def sec_to_str(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label Training (Stage 1 - Label-Level Dynamic FkL + Early Cutting)')
    parser.add_argument('--dataname', help='dataname', default='nih', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot')
    parser.add_argument('--img_size', default=448, type=int, help='size of input images')

    parser.add_argument('--output', metavar='DIR', help='path to output folder')
    parser.add_argument('--num_class', default=14, type=int, help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--optim', default='SGD', type=str, choices=['AdamW', 'Adam_twd', 'SGD'])
    
    parser.add_argument('--scheduler', default='OneCycle', type=str, choices=['OneCycle', 'StepLR'])
    parser.add_argument('--step_size', default=40, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    
    # Loss args
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--dtgfl', action='store_true', default=False)              
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--gamma_neg', default=2, type=float)
    parser.add_argument('--loss_dev', default=-1, type=float)
    parser.add_argument('--loss_clip', default=0.0, type=float)  

    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')

    parser.add_argument('--ema-decay', default=0.9997, type=float)
    parser.add_argument('--ema-epoch', default=0, type=int)
    parser.add_argument('--seed', default=95, type=int)

    # Transformer params
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=8192, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--keep_other_self_attn_dec', action='store_true')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true')
    parser.add_argument('--keep_input_proj', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False)
    
    # ================= 核心对齐超参数 (ICCV 2023 & NeurIPS 2025) =================
    parser.add_argument('--splicemix_start_epoch', default=10, type=int, help='执行 Early Cutting 截断并退出的 Epoch 节点')
    
    # ICCV 2023 核心对齐参数
    parser.add_argument('--fkl_warmup_epochs', default=6, type=int, help='严格对齐代码 WARM_UP_EPOCHS=6')
    parser.add_argument('--fkl_consecutive_epochs', default=3, type=int, help='对齐 lista, list: 需要连续 3 个 epoch 标签预测一致')
    
    # NeurIPS 2025 MEE 核心对齐参数
    parser.add_argument('--early_cutting_rate', default=1.5, type=float, help='MEE 截断比例系数 early_cutting_rate=1.5')
    parser.add_argument('--newremove_rate', default=90000, type=int, help='标签级候选截断最大数量 (相当于样本级3000 * 10类)')
    parser.add_argument('--top_conf_ratio', default=0.2, type=float, help='对齐 MEE top_conf_ratio=0.2 (前20%)')
    parser.add_argument('--low_grad_ratio', default=0.2, type=float, help='对齐 MEE low_grad_ratio=0.2 (后20%)')
    
    # 优化器严格对齐
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', default=5e-4, type=float, dest='weight_decay', help='严格对齐论文 weight_decay=5e-4')
    # =============================================================================
    parser.add_argument('--orid_norm', action='store_true', help='Use original dataset normalization')
    parser.add_argument('-cd', '--cuda_devices', default=[0], nargs='+', type=int)
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    args.world_size = 1
    gpu_id = args.cuda_devices[0] if hasattr(args, 'cuda_devices') else 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0) 
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="Stage1-LabelLevel-MEE")
    logger.info("Command: "+' '.join(sys.argv))
    
    with open(os.path.join(args.output, "config.json"), 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)

    return main_worker(args, logger)

def main_worker(args, logger):
    model = build_q2l(args).cuda()
    ema_m = ModelEma(model, args.ema_decay) 

    criterion = AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, 
        clip=args.loss_clip, disable_torch_grad_focal_loss=args.dtgfl
    )

    base_lr = args.lr
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'backbone' in name or 'fc_splicemix' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
            
    param_dicts = [{"params": backbone_params, "lr": base_lr*0.1}, {"params": other_params, "lr": base_lr}]
    optimizer = torch.optim.SGD(param_dicts, lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    train_dataset, val_dataset = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 构建无数据增强的 FkL 追踪 Loader，保证预测特征的稳定性
    import copy
    train_dataset_eval = copy.deepcopy(train_dataset)
    train_dataset_eval.transform = val_dataset.transform 
    etrain_loader = torch.utils.data.DataLoader(train_dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=[g['lr'] for g in optimizer.param_groups], steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.5)

    # ================= 阶段一：标签级 FkL 追踪矩阵初始化 =================
    num_train_samples = len(train_dataset)
    fkl_consecutive_counts = torch.zeros((num_train_samples, args.num_class), dtype=torch.int32).cuda()
    fkl_mask = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).cuda()
    
    # 【新增】: 提取并全局保存所有样本的原始 Target 矩阵 (用于 Stage 3 过滤)
    all_targets = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).cuda()
    for _, targets, indices in etrain_loader:
        all_targets[indices] = (targets == 1).cuda()
    
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
            
        model.train()
        train_loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args)
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        # --- 【阶段一核心】: LateStopping 时间维度追踪困难样本 ---
        if epoch >= args.fkl_warmup_epochs:
            logger.info("    -> Tracking LABEL-LEVEL predictions for FkL...")
            ema_m.module.eval()
            with torch.no_grad():
                for images, targets, indices in etrain_loader:
                    images, targets = images.cuda(), targets.cuda()
                    indices = indices.cuda()
                    
                    logits, _, _, _ = ema_m.module(images)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    
                    # 矩阵维度: (Batch, 14). 判断单个标签预测是否等于目标
                    is_correct = (preds == targets)
                    
                    counts = fkl_consecutive_counts[indices]
                    # ICCV2023 逻辑: 连续正确+1，只要预测错误一次直接归零重置
                    counts = torch.where(is_correct, counts + 1, torch.zeros_like(counts))
                    fkl_consecutive_counts[indices] = counts
                    
                    # 只要连续正确次数达到阈值(默认3次)，就将其标为候选干净标签(FkL)
                    new_fkl = (counts >= args.fkl_consecutive_epochs)
                    fkl_mask[indices] = fkl_mask[indices] | new_fkl

        metrics_res, val_loss = validate(val_loader, ema_m.module, criterion, args)
        logger.info(f"    -> EMA Model Val mAUC: {metrics_res['mAUC']:.4f}")

        # --- 【阶段二 & 三 & 四】: Early Cutting 剔除与最终聚合输出 ---
        if epoch == args.splicemix_start_epoch:
            logger.info(f"\n>>> [Stage 1 & 2 & 3] Epoch {epoch} Reached. Executing Label-Level Early Cutting & Aggregation... <<<")
            
            # 【阶段二】: Early Cutting 提纯 (排雷简单噪声标签 MEE)
            clean_label_mask = perform_label_level_early_cutting(
                model=ema_m.module, 
                dataset=train_dataset_eval, 
                fkl_mask=fkl_mask, 
                args=args, 
                logger=logger
            )
            
            # ====================================================================
            # 【阶段三】: 标签到样本的聚合投票 (仅基于 Target == 1 的标签)
            # ====================================================================
            
            # 1. 统计每个样本中，标注为 1 (阳性疾病) 的标签总数
            total_positives = all_targets.sum(dim=1) 
            
            # 2. 统计标注为 1 且被【阶段二】判定为干净的标签数量
            clean_pos_counts = (clean_label_mask & all_targets).sum(dim=1)
            
            # 3. 统计标注为 1 且被【阶段二】判定为噪声的标签数量
            noisy_pos_counts = (~clean_label_mask & all_targets).sum(dim=1)
            
            # 4. 样本级判别规则初始化
            is_clean_sample = torch.zeros(num_train_samples, dtype=torch.bool).cuda()
            
            # 情况 A：对于有疾病标注的样本（至少有一个 Target == 1）
            # 判别核心：正标签中，干净数量 > 噪声数量
            has_pos = total_positives > 0
            is_clean_sample[has_pos] = clean_pos_counts[has_pos] > noisy_pos_counts[has_pos]
            
            # 情况 B：对于完全健康的样本（Target 全为 0）
            # 物理意义：无病灶标注极其可靠，直接无条件视为干净样本！
            no_pos = total_positives == 0
            is_clean_sample[no_pos] = True
            
            # 提取最终的样本级索引
            clean_indices = torch.nonzero(is_clean_sample).squeeze(1).tolist()
            noisy_indices = torch.nonzero(~is_clean_sample).squeeze(1).tolist()
            # 【阶段四】: 为噪声样本生成 "纯净标签字典" 供 CAM 使用
            noise_clean_labels_dict = {}
            for n_idx in noisy_indices:
                # 提取该噪声样本中，仍然被判定为干净的标签索引 [0, 1, ..., 13]
                clean_lbls = torch.nonzero(clean_label_mask[n_idx]).squeeze(1).tolist()
                noise_clean_labels_dict[int(n_idx)] = clean_lbls
                
            # 落盘保存为 stage 2 及 CAM 提供所需格式
            torch.save(clean_indices, os.path.join(args.output, 'clean_indices.pt'))
            torch.save(noisy_indices, os.path.join(args.output, 'noisy_indices.pt'))
            torch.save(noise_clean_labels_dict, os.path.join(args.output, 'noise_clean_labels_dict.pt'))
            
            logger.info("="*60)
            logger.info(f" >>> Stage 1 Split Complete! ")
            logger.info(f"     * Clean Samples: {len(clean_indices)}")
            logger.info(f"     * Noisy Samples: {len(noisy_indices)}")
            logger.info(f"     * Saved `noise_clean_labels_dict.pt` mapping noisy samples to their clean tags.")
            logger.info(" >>> Exiting Stage 1 gracefully to start Stage 2 / CAM scripts. <<<")
            logger.info("="*60)
            sys.stdout.flush() 
            sys.exit(0)

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    model.train()
    total_loss = 0.0
    for images, target, _ in train_loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out_trans_all, _, _, _ = model(images)
            loss = criterion(out_trans_all, target)
        
        if loss.item() > 0:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        if epoch >= args.ema_epoch:
            ema_m.update(model)
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def validate(val_loader, model, criterion, args): 
    meter = AveragePrecisionMeter(difficult_examples=False)
    model.eval()
    total_loss = 0
    for images, target, _ in val_loader: 
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output, _, _, _ = model(images)
            loss = criterion(output, target)
            total_loss += loss.item()
            meter.add(output, target, filename=[]) 
    return meter.compute_all_metrics(), total_loss / len(val_loader)

# ==================== 【阶段二】标签级 Early Cutting 逻辑 ====================
def perform_label_level_early_cutting(model, dataset, fkl_mask, args, logger):
    """
    专门针对多标签分类设计的 MEE Early Cutting (NeurIPS 2025 严格对齐)
    """
    model.eval()
    # 限制 bs 以防求梯度时 OOM
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    num_samples = len(dataset)
    num_classes = args.num_class
    
    loss_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).cuda()
    conf_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).cuda()
    grad_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).cuda()
    
    logger.info("    -> Calculating Loss, Confidence, and Gradient Norm across all samples...")
    
    for images, targets, indices in loader:
        images, targets = images.cuda(), targets.cuda()
        indices = indices.cuda()
        images.requires_grad_(True)
        
        logits, _, _, _ = model(images)
        probs = torch.sigmoid(logits)
        
        # 1. BCE 标签级 Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. 标签级绝对置信度 (偏离0.5的程度)
        conf = torch.abs(probs - 0.5)
        
        # 3. 标签级 Grad Norm 近似
        sample_losses = bce_loss.mean(dim=1)
        grads = torch.autograd.grad(sample_losses.sum(), images)[0]
        # (Batch_size, 1) -> 广播到 14 类
        g_norms = torch.norm(grads.flatten(1), dim=1).unsqueeze(1).expand(-1, num_classes)
        
        loss_tensor[indices] = bce_loss.detach()
        conf_tensor[indices] = conf.detach()
        grad_tensor[indices] = g_norms.detach()

    # 获取所有 FkL 候选标签的位置坐标 [M, 2]
    fkl_indices = torch.nonzero(fkl_mask) 
    M = len(fkl_indices)
    logger.info(f"    -> Found {M} FkL label candidates out of {num_samples * num_classes} total labels.")
    
    if M > 0:
        fkl_losses = loss_tensor[fkl_mask]
        fkl_confs = conf_tensor[fkl_mask]
        fkl_grads = grad_tensor[fkl_mask]
        
        # Loss Ranking: 取出 Loss 最高的疑似 MEE 标签 [严格对齐 early_cutting_rate = 1.5]
        num_candidates = int(M / args.early_cutting_rate)
        num_candidates = min(args.newremove_rate, num_candidates)
        
        _, sorted_idx = torch.sort(fkl_losses, descending=True)
        cand_idx = sorted_idx[:num_candidates] 
        
        cand_confs = fkl_confs[cand_idx]
        cand_grads = fkl_grads[cand_idx]
        
        if len(cand_idx) > 0:
            # 严格对齐 MEE: 置信度前20%，梯度范数后20%
            conf_thresh = torch.quantile(cand_confs, 1.0 - args.top_conf_ratio)
            grad_thresh = torch.quantile(cand_grads, args.low_grad_ratio)
            
            is_mee = (cand_confs >= conf_thresh) & (cand_grads <= grad_thresh)
            mee_local_idx = cand_idx[is_mee]
            
            mee_count = len(mee_local_idx)
            logger.info(f"    -> MEE Refinement: Conf>=Top {args.top_conf_ratio*100}%, Grad<=Bottom {args.low_grad_ratio*100}%")
            logger.info(f"    -> Final MEE Labels CUT from Clean pool: {mee_count}")
            
            if mee_count > 0:
                # 映射回全局矩阵坐标，将其从干净集(True) 踢出为 噪声/MEE (False)
                mee_global_indices = fkl_indices[mee_local_idx]
                fkl_mask[mee_global_indices[:, 0], mee_global_indices[:, 1]] = False
                
    return fkl_mask # 返回净化后的 Clean Label Mask 矩阵 [N, 14]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

if __name__ == '__main__':
    main()