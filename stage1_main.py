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
    parser = argparse.ArgumentParser(description='Query2Label NIH Training (Stage 1 - MEE Label-Level)')
    # == 原有基础参数 ==
    parser.add_argument('--dataname', default='nih', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', default='/comp_robot')
    parser.add_argument('--img_size', default=448, type=int)
    parser.add_argument('--output', metavar='DIR', help='path to output folder')
    parser.add_argument('--num_class', default=14, type=int)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--optim', default='SGD', type=str, choices=['AdamW', 'SGD'])
    parser.add_argument('--scheduler', default='OneCycle', type=str)
    parser.add_argument('--step_size', default=40, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    
    parser.add_argument('--eps', default=1e-5, type=float)
    parser.add_argument('--dtgfl', action='store_true', default=False)              
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--gamma_neg', default=2, type=float)
    parser.add_argument('--loss_clip', default=0.0, type=float)  

    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--seed', default=95, type=int)

    # == Q2L Transformer 参数 ==
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dim_feedforward', default=8192, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    
    # ================= 标签级 MEE 核心对齐参数 =================
    parser.add_argument('--splicemix_start_epoch', default=10, type=int, help='Epoch 截断节点 (Stage 1 退出点)')
    parser.add_argument('--warm_up_epochs', default=6, type=int, help='前几个 Epoch 不记录 FkL')
    parser.add_argument('--fkl_consecutive_epochs', default=3, type=int, help='需要连续多少次预测正确才算候选干净标签')
    
    parser.add_argument('--early_cutting_rate', default=1.5, type=float)
    parser.add_argument('--newremove_rate', default=90000, type=int, help='标签级候选截断最大数量')
    parser.add_argument('--top_conf_ratio', default=0.2, type=float)
    parser.add_argument('--low_grad_ratio', default=0.2, type=float)
    
    # 单卡设备
    parser.add_argument('-cd', '--cuda_devices', default=[0], nargs='+', type=int)
    parser.add_argument('--orid_norm', action='store_true', default=False)
    
    args = parser.parse_args()
    return args

class ModelEMA:
    def __init__(self, model, alpha=0.999):
        self.model = model
        self.ema_model = deepcopy(model)
        self.alpha = alpha
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def step(self):
        a = self.alpha
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(a).add_(p, alpha=1.0 - a)
        for b, ema_b in zip(self.model.buffers(), self.ema_model.buffers()):
            ema_b.copy_(b)

def build_dual_optimizers(net1, net2, args):
    def get_params(model):
        backbone, other = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'backbone' in name: backbone.append(param)
            else: other.append(param)
        return [{"params": backbone, "lr": args.lr * 0.1}, {"params": other, "lr": args.lr}]
    
    opt1 = torch.optim.SGD(get_params(net1), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    opt2 = torch.optim.SGD(get_params(net2), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return opt1, opt2

def ensemble_logits(models, x):
    net1, net2, net1e, net2e = models
    out1, _, _, _ = net1(x)
    out2, _, _, _ = net2(x)
    out1e, _, _, _ = net1e(x)
    out2e, _, _, _ = net2e(x)
    return (out1 + out2 + out1e + out2e) / 4.0

def train_one_epoch_mutual_kd(net1, net2, net1e, net2e, ema1, ema2, opt1, opt2, loader, criterion, args, device, global_label_mask):
    net1.train()
    net2.train()
    temperature = 2.0
    alpha = 0.5
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # 注意：这里 loader 必须能返回 indices，用于索引 global_label_mask
    for images, targets, indices in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        
        # 获取当前 batch 在全局矩阵中被判定为干净的标签掩码 [Batch, 14]
        batch_mask = global_label_mask[indices].float()

        with torch.no_grad():
            t1, _, _, _ = net1(images)
            t2, _, _, _ = net2(images)
            t1e, _, _, _ = net1e(images)
            t2e, _, _, _ = net2e(images)
            
            soft1 = torch.sigmoid(t1 / temperature)
            soft2 = torch.sigmoid(t2 / temperature)
            soft1e = torch.sigmoid(t1e / temperature)
            soft2e = torch.sigmoid(t2e / temperature)
            
            soft1_comb = (soft1 + soft1e) / 2.0
            soft2_comb = (soft2 + soft2e) / 2.0

        with torch.cuda.amp.autocast(enabled=args.amp):
            z1, _, _, _ = net1(images)
            z2, _, _, _ = net2(images)
            
            # 使用 BCE 计算 Hard Loss，保持维度 [Batch, 14]，并用 batch_mask 屏蔽掉脏标签
            hard1 = F.binary_cross_entropy_with_logits(z1, targets.float(), reduction='none')
            hard2 = F.binary_cross_entropy_with_logits(z2, targets.float(), reduction='none')
            
            # Soft Loss 同样屏蔽
            soft1_loss = F.binary_cross_entropy_with_logits(z1 / temperature, soft2_comb, reduction='none') * (temperature ** 2)
            soft2_loss = F.binary_cross_entropy_with_logits(z2 / temperature, soft1_comb, reduction='none') * (temperature ** 2)
            
            # 聚合总 Loss，只对 Mask=1 的干净标签求均值
            loss1 = ((hard1 + alpha * soft1_loss) * batch_mask).sum() / max(1.0, batch_mask.sum().item())
            loss2 = ((hard2 + alpha * soft2_loss) * batch_mask).sum() / max(1.0, batch_mask.sum().item())

        opt1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.step(opt1)
        ema1.step()

        opt2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.step(opt2)
        ema2.step()
        scaler.update()
@torch.inference_mode()
def update_fkl_mask(models, loader, device, fkl_consecutive_counts, fkl_mask, consecutive_epochs_required):
    """
    更新标签级 FkL 矩阵: 维度 [N, 14]
    """
    for x, y, indices in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        
        logits = ensemble_logits(models, x)
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        # 判断每个标签是否预测正确 [Batch, 14]
        is_correct = (preds == y)
        
        counts = fkl_consecutive_counts[indices]
        # 如果预测正确，连续次数+1；一旦错误，直接归零
        counts = torch.where(is_correct, counts + 1, torch.zeros_like(counts))
        fkl_consecutive_counts[indices] = counts
        
        # 只要连续正确次数达到阈值，就标为候选干净标签
        new_fkl = (counts >= consecutive_epochs_required)
        fkl_mask[indices] = fkl_mask[indices] | new_fkl


def perform_label_level_early_cutting(models, dataset, fkl_mask, args, logger, device):
    """
    标签级 MEE 截断 (防 OOM 优化版)
    """
    for m in models: m.eval()
    eval_bs = max(1, args.batch_size // 2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=eval_bs, shuffle=False, num_workers=args.workers)
    
    num_samples = len(dataset)
    num_classes = args.num_class
    
    loss_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).to(device)
    conf_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).to(device)
    grad_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).to(device)
    
    logger.info("    -> Calculating Loss, Confidence, and Gradient Norm across all samples for Early Cutting...")
    
    for images, targets, indices in loader:
        images, targets = images.to(device), targets.to(device)
        indices = indices.to(device)
        images.requires_grad_(True)
        
        logits = ensemble_logits(models, images)
        probs = torch.sigmoid(logits)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        conf = torch.abs(probs - 0.5)  # 偏离 0.5 越远代表置信度越高
        
        total_loss = bce_loss.sum()
        grads = torch.autograd.grad(total_loss, images)[0]
        g_norms = torch.norm(grads.flatten(1), dim=1).unsqueeze(1).expand(-1, num_classes)
        
        loss_tensor[indices] = bce_loss.detach()
        conf_tensor[indices] = conf.detach()
        grad_tensor[indices] = g_norms.detach()

        images.requires_grad_(False)
        del grads, total_loss, logits, probs, bce_loss
        torch.cuda.empty_cache()

    fkl_indices = torch.nonzero(fkl_mask) 
    M = len(fkl_indices)
    logger.info(f"    -> Found {M} FkL label candidates out of {num_samples * num_classes} total labels.")
    
    if M > 0:
        fkl_losses = loss_tensor[fkl_mask]
        fkl_confs = conf_tensor[fkl_mask]
        fkl_grads = grad_tensor[fkl_mask]
        
        num_candidates = int(M / args.early_cutting_rate)
        num_candidates = min(args.newremove_rate, num_candidates)
        
        _, sorted_idx = torch.sort(fkl_losses, descending=True)
        cand_idx = sorted_idx[:num_candidates] 
        
        cand_confs = fkl_confs[cand_idx]
        cand_grads = fkl_grads[cand_idx]
        
        if len(cand_idx) > 0:
            conf_thresh = torch.quantile(cand_confs, 1.0 - args.top_conf_ratio)
            grad_thresh = torch.quantile(cand_grads, args.low_grad_ratio)
            
            is_mee = (cand_confs >= conf_thresh) & (cand_grads <= grad_thresh)
            mee_local_idx = cand_idx[is_mee]
            
            mee_count = len(mee_local_idx)
            logger.info(f"    -> MEE Refinement Thresholds: Conf >= {conf_thresh:.4f}, Grad <= {grad_thresh:.4f}")
            logger.info(f"    -> Final MEE Labels CUT from Clean pool: {mee_count}")
            
            if mee_count > 0:
                mee_global_indices = fkl_indices[mee_local_idx]
                fkl_mask[mee_global_indices[:, 0], mee_global_indices[:, 1]] = False
                
    return fkl_mask 
def pick_remove_rate_by_phase(rn, i1, i2, i3, r1, r2, r3, r4):
    """根据当前的 Round 轮次，匹配对应的 remove_rate"""
    if rn <= i1:
        return r1
    if rn <= i1 + i2:
        return r2
    if rn <= i1 + i2 + i3:
        return r3
    return r4
def main():
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_devices[0])
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    cudnn.benchmark = True

    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="Stage1-MEE-LabelLevel")
    logger.info("Command: " + ' '.join(sys.argv))

    train_dataset, val_dataset = get_datasets(args)
    # 无增强测试 Loader
    train_dataset_eval = deepcopy(train_dataset)
    train_dataset_eval.transform = val_dataset.transform

    criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.loss_clip, disable_torch_grad_focal_loss=args.dtgfl)

    # 提取全局 Target 矩阵，用于最终的 Clean > Noisy 投票
    etrain_loader = torch.utils.data.DataLoader(train_dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    num_train_samples = len(train_dataset)
    all_targets = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)
    for _, targets, indices in etrain_loader:
        all_targets[indices] = (targets == 1).to(device)

    # 标签级 FkL 矩阵初始化
    fkl_consecutive_counts = torch.zeros((num_train_samples, args.num_class), dtype=torch.int32).to(device)
    fkl_mask = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)

    logger.info("=================== Stage 1 Training Start ===================")
    # === 初始化全局掩码 ===
    # 初始状态下，所有标签都视为“干净”（参与训练）
    global_label_mask = torch.ones((num_train_samples, args.num_class), dtype=torch.bool).to(device)
    num_rounds = getattr(args, 'num_rounds', 3) 
    fkl_threshold = getattr(args, 'fkl_threshold', args.newremove_rate)
    logger.info(f"=================== Stage 1 Training Start (Multi-Round Mode: {num_rounds} Rounds) ===================")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    # ================= 修改三：真正激活 Multi-Round 逻辑 =================
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n" + "="*20 + f" Starting Round {round_num}/{num_rounds} " + "="*20)
        
        # 可选：根据当前 Round 动态调整本轮的 remove_rate
        # args.newremove_rate = pick_remove_rate_by_phase(round_num, 1, 1, 1, 20000, 40000, 60000, 80000)
        
        # ================= 关键修改：每轮重新初始化网络与优化器 =================
        # 确保模型不受上一轮脏数据梯度的影响 (Cold Start)
        net1 = build_q2l(args).to(device)
        net2 = build_q2l(args).to(device)
        ema1 = ModelEMA(net1, alpha=0.999)
        ema2 = ModelEMA(net2, alpha=0.999)
        models = (net1, net2, ema1.ema_model, ema2.ema_model)

        opt1, opt2 = build_dual_optimizers(net1, net2, args)
        
        # ================= 关键修改：每轮重置 FkL 统计矩阵 =================
        # 因为网络重置了，连续正确预测的次数也必须重新从 0 开始计算
        fkl_consecutive_counts = torch.zeros((num_train_samples, args.num_class), dtype=torch.int32).to(device)
        fkl_mask = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)
        for epoch in range(args.epochs):
            logger.info(f"\n[Round {round_num}/{num_rounds}] Epoch {epoch + 1}/{args.epochs}")
            
            # 1. 训练一个 Epoch (现在正确传入了 global_label_mask)
            train_one_epoch_mutual_kd(net1, net2, ema1.ema_model, ema2.ema_model, ema1, ema2, opt1, opt2, train_loader, criterion, args, device, global_label_mask)
            
            # 2. 追踪标签级 FkL
            if epoch >= args.warm_up_epochs:
                logger.info("    -> Tracking LABEL-LEVEL predictions for FkL...")
                update_fkl_mask(models, etrain_loader, device, fkl_consecutive_counts, fkl_mask, args.fkl_consecutive_epochs)
                
                # 统计当前满足连续预测正确的 FkL 候选数量
                current_fkl_count = fkl_mask.sum().item()
                logger.info(f"    -> Current FkL Candidates: {current_fkl_count} (Threshold: {fkl_threshold})")
                
                # === 核心逻辑：判断是否达到截断条件 ===
                # 如果候选数量达标，或者达到了当前 Round 设置的强行退出 Epoch
                if current_fkl_count >= fkl_threshold or epoch == args.splicemix_start_epoch:
                    logger.info(f"\n>>> Condition Met! Executing Label-Level Early Cutting for Round {round_num}... <<<")
                    
                    # 【阶段A】剔除高Loss、高置信度、低梯度的 MEE 噪声标签，返回更新后的掩码
                    clean_label_mask = perform_label_level_early_cutting(models, train_dataset_eval, fkl_mask, args, logger, device)
                    
                    # 更新全局掩码，供下一轮（或者后面的投票）使用
                    global_label_mask = clean_label_mask
                    
                    # 判断是否是最后一轮
                    if round_num == num_rounds:
                        logger.info(f"\n>>> Final Round {round_num} Reached. Executing Sample-Level Voting & Exiting! <<<")
                        
                        # 【阶段B】样本级聚合投票 (核心: clean_pos > noisy_pos)
                        total_positives = all_targets.sum(dim=1) 
                        clean_pos_counts = (clean_label_mask & all_targets).sum(dim=1)
                        noisy_pos_counts = (~clean_label_mask & all_targets).sum(dim=1)
                        
                        is_clean_sample = torch.zeros(num_train_samples, dtype=torch.bool).to(device)
                        
                        # 规则 1：有疾病标注的样本，干净正标签 > 噪声正标签
                        has_pos = total_positives > 0
                        is_clean_sample[has_pos] = clean_pos_counts[has_pos] > noisy_pos_counts[has_pos]
                        
                        # 规则 2：完全健康的样本（Target 全为 0），无条件视为干净样本
                        no_pos = total_positives == 0
                        is_clean_sample[no_pos] = True
                        
                        clean_indices = torch.nonzero(is_clean_sample).squeeze(1).tolist()
                        noisy_indices = torch.nonzero(~is_clean_sample).squeeze(1).tolist()
                        
                        # 生成噪声样本的干净标签字典，供 Stage 2 CAM 使用
                        noise_clean_labels_dict = {}
                        for n_idx in noisy_indices:
                            clean_lbls = torch.nonzero(clean_label_mask[n_idx]).squeeze(1).tolist()
                            noise_clean_labels_dict[int(n_idx)] = clean_lbls

                        # 保存并退出
                        torch.save(clean_indices, os.path.join(args.output, 'clean_indices.pt'))
                        torch.save(noisy_indices, os.path.join(args.output, 'noisy_indices.pt'))
                        torch.save(noise_clean_labels_dict, os.path.join(args.output, 'noise_clean_labels_dict.pt'))
                        
                        logger.info("="*60)
                        logger.info(f" >>> Stage 1 Split Complete! ")
                        logger.info(f"     * Clean Samples: {len(clean_indices)}")
                        logger.info(f"     * Noisy Samples: {len(noisy_indices)}")
                        logger.info(" >>> Exiting Stage 1 gracefully to start Stage 2 / CAM scripts. <<<")
                        logger.info("="*60)
                        
                        sys.stdout.flush() 
                        sys.exit(0)
                    else:
                        # 如果不是最后一轮，跳出当前的 epoch 循环，直接进入下一个 Round
                        logger.info(f">>> Round {round_num} Completed. Breaking inner loop to start Round {round_num + 1}. <<<")
                        break  # 打断 `for epoch in range(args.epochs):` 循环

if __name__ == '__main__':
    main()