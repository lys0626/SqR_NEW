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
import csv
from lib.dataset.get_dataset import get_datasets
from lib.utils.logger import setup_logger
from lib.models.query2label import build_q2l
from lib.utils.metric import AveragePrecisionMeter
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict
from collections import defaultdict
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
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--output', metavar='DIR', help='path to output folder')
    parser.add_argument('--num_class', default=14, type=int)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'SGD'])
    parser.add_argument('--scheduler', default='OneCycle', type=str)
    parser.add_argument('--step_size', default=40, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    
    parser.add_argument('--eps', default=1e-5, type=float)           
    parser.add_argument('--gamma_pos', default=0, type=float)
    parser.add_argument('--gamma_neg', default=4, type=float)
    parser.add_argument('--loss_clip', default=0.05, type=float)  

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
    parser.add_argument('--keep_other_self_attn_dec', action='store_true')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true')
    parser.add_argument('--keep_input_proj', action='store_true')
    # ================= 标签级 MEE 核心对齐参数 =================
    parser.add_argument('--warm_up_epochs', default=6, type=int, help='前几个 Epoch 不记录 FkL')
    parser.add_argument('--fkl_consecutive_epochs', default=3, type=int, help='需要连续多少次预测正确才算候选干净标签')
    
    parser.add_argument('--early_cutting_rate', default=1.5, type=float)
    parser.add_argument('--newremove_rate', default=90000, type=int, help='标签级候选截断最大数量')
    parser.add_argument('--top_conf_ratio', default=0.2, type=float)
    parser.add_argument('--low_grad_ratio', default=0.2, type=float)
    
    # 单卡设备
    parser.add_argument('-cd', '--cuda_devices', default=[0], nargs='+', type=int)
    parser.add_argument('--orid_norm', action='store_true', default=False)

    # Phase 控制参数
    parser.add_argument("--i_rate_1", type=int, default=3)
    parser.add_argument("--i_rate_2", type=int, default=3)
    parser.add_argument("--i_rate_3", type=int, default=3)
    parser.add_argument("--i_rate_4", type=int, default=0)
    
    # 动态保留比例 (注意：在 MEE 中 remove_rate 其实是 "保留的干净样本比例")
    parser.add_argument("--remove_rate_1", type=float, default=0.995)
    parser.add_argument("--remove_rate_2", type=float, default=0.995)
    parser.add_argument("--remove_rate_3", type=float, default=0.995)
    parser.add_argument("--remove_rate_4", type=float, default=0.995)
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
    
    if args.optim == 'AdamW':
        opt1 = torch.optim.AdamW(get_params(net1), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
        opt2 = torch.optim.AdamW(get_params(net2), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
        
    elif args.optim == 'SGD':
        opt1 = torch.optim.SGD(get_params(net1), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        opt2 = torch.optim.SGD(get_params(net2), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
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
            
            # # 使用 BCE 计算 Hard Loss，保持维度 [Batch, 14]，并用 batch_mask 屏蔽掉脏标签
            # hard1 = F.binary_cross_entropy_with_logits(z1, targets.float(), reduction='none')
            # hard2 = F.binary_cross_entropy_with_logits(z2, targets.float(), reduction='none')
            # soft1_loss = F.binary_cross_entropy_with_logits(z1 / temperature, soft2_comb, reduction='none') * (temperature ** 2)
            # soft2_loss = F.binary_cross_entropy_with_logits(z2 / temperature, soft1_comb, reduction='none') * (temperature ** 2)
            # # 聚合总 Loss，只对 Mask=1 的干净标签求均值
            # loss1 = ((hard1 + alpha * soft1_loss) * batch_mask).sum() / max(1.0, batch_mask.sum().item())
            # loss2 = ((hard2 + alpha * soft2_loss) * batch_mask).sum() / max(1.0, batch_mask.sum().item())



            # ==========================================================
            # 1. Hard Loss 区域：使用 ASL + Hack Mask (对抗标签不平衡)
            # ==========================================================
            masked_z1 = z1.clone()
            masked_z2 = z2.clone()
            
            ignore_idx = (batch_mask == 0)
            
            # 把被认定为脏标签的位置，强行改成符合 target 的极值，使其 ASL Loss 为 0
            masked_z1[ignore_idx] = 20.0
            masked_z2[ignore_idx] = 20.0

            # 真正调用 ASL 计算 Hard Loss！
            hard1 = criterion(masked_z1, targets.float())
            hard2 = criterion(masked_z2, targets.float())
            # Soft Loss 同样屏蔽
            soft1_loss = F.binary_cross_entropy_with_logits(z1 / temperature, soft2_comb, reduction='none') * (temperature ** 2)
            soft2_loss = F.binary_cross_entropy_with_logits(z2 / temperature, soft1_comb, reduction='none') * (temperature ** 2)

            # ==========================================================
            # 3. 聚合总 Loss
            # ==========================================================
            # hard 已经是处理过 mask 的均值标量了
            # soft 还是 [Batch, 14] 矩阵，需要乘 batch_mask 屏蔽脏标签后求均值
            loss1 = hard1 + alpha * (soft1_loss * batch_mask).sum() / max(1.0, batch_mask.sum().item())
            loss2 = hard2 + alpha * (soft2_loss * batch_mask).sum() / max(1.0, batch_mask.sum().item())

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
        preds = (torch.sigmoid(logits) > 0.2).float()
        
        # ==========================================================
        # 🌟 核心修改：只统计原本标注为 1 的正标签
        # ==========================================================
        # 条件 1: 预测正确 (preds == y)
        # 条件 2: 真实标签为正类 (y == 1)
        is_correct_and_positive = (preds == y) & (y == 1)
        
        counts = fkl_consecutive_counts[indices]
        # 如果预测正确，连续次数+1；一旦错误，直接归零
        counts = torch.where(is_correct_and_positive, counts + 1, torch.zeros_like(counts))
        fkl_consecutive_counts[indices] = counts
        
        # 只要连续正确次数达到阈值，就标为候选干净标签
        new_fkl = (counts >= consecutive_epochs_required)
        fkl_mask[indices] = fkl_mask[indices] | new_fkl




def perform_label_level_early_cutting(models, dataset, fkl_mask, args, logger, device,all_targets):
    """
    严格对齐原版 MEE 逻辑：高 Loss (候选) -> 高置信度 + 低输入梯度范数 (确诊噪声)
    """
    for m in models: m.eval()
    
    num_samples = len(dataset)
    num_classes = args.num_class
    
    # ==========================================================
    # 阶段 1：全局无梯度扫一遍，寻找 High Loss 候选标签
    # ==========================================================
    logger.info("    -> Step 1: Calculating BCE Loss across FkL candidates (No Grad)...")
    eval_bs = max(1, args.batch_size // 2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=eval_bs, shuffle=False, num_workers=args.workers)
    
    loss_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32).to(device)
    
    for images, targets, indices in loader:
        images, targets = images.to(device), targets.to(device)
        indices = indices.to(device)
        with torch.no_grad():
            logits = ensemble_logits(models, images)
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
            loss_tensor[indices] = bce_loss.detach()

    fkl_indices = torch.nonzero(fkl_mask) # [M, 2], 每一行是 [sample_idx, class_idx]
    # 提取全局 Target 矩阵中原本标注为 1 的位置
    positive_targets = (all_targets == 1)

    # 只计算那些原本标注为 1 的标签里的 FkL
    fkl_indices = torch.nonzero(fkl_mask & positive_targets)
    M = len(fkl_indices) 
    # 此时的 M 才是真正有意义的“疾病候选者”，通常只有几万的数量级
    logger.info(f"    -> Found {M} FkL label candidates.")

    if M == 0:
        return fkl_mask, torch.zeros_like(fkl_mask)

    # 提取 FkL 对应的 Loss 并排序，截断获取 num_candidates
    fkl_losses = loss_tensor[fkl_mask]
    num_candidates = int(M / args.early_cutting_rate)
    num_candidates = min(args.newremove_rate, num_candidates)
    
    _, sorted_idx = torch.sort(fkl_losses, descending=True)
    cand_local_idx = sorted_idx[:num_candidates] 
    cand_global_indices = fkl_indices[cand_local_idx] # 截取出真正的候选标签索引 [num_candidates, 2]
    
    logger.info(f"    -> Step 2: Calculating Confidence and INPUT GRADIENT NORM for {num_candidates} specific labels...")

    # ==========================================================
    # 阶段 2：仅对候选样本开启梯度计算，严格提取 Input Gradient Norm
    # ==========================================================
    # 构建候选样本的字典映射：sample_idx -> [(class_idx, global_array_index), ...]
    sample_to_cand_classes = defaultdict(list)
    for i, (s_idx, c_idx) in enumerate(cand_global_indices.cpu().numpy()):
        sample_to_cand_classes[s_idx].append((c_idx, i))

    # 构建专属的 Subset DataLoader，避免对无关图片进行前向传播
    unique_sample_indices = list(sample_to_cand_classes.keys())
    subset = torch.utils.data.Subset(dataset, unique_sample_indices)
    
    # ==========================================================
    # 🌟 核心修复：强制缩小 Step 2 的 Batch Size，并清空显存碎片
    # ==========================================================
    torch.cuda.empty_cache()  # 释放 Step 1 遗留的无用显存
    
    grad_bs = 8  # ⚠️ 针对 24G 显卡 + 4模型集成的安全阈值 (若显存更大可设为 16)
    
    logger.info(f"    -> [Memory Protector] Re-building subset loader with ultra-small batch size: {grad_bs}")
    
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=grad_bs, shuffle=False, num_workers=args.workers)
    # ==========================================================

    conf_arr = torch.zeros(num_candidates, dtype=torch.float32).to(device)
    grad_arr = torch.zeros(num_candidates, dtype=torch.float32).to(device)

    for images, targets, indices in subset_loader:
        images, targets = images.to(device), targets.to(device)
        indices = indices.numpy()
        
        # 【关键】：开启输入图片的梯度追踪
        images.requires_grad_(True)
        logits = ensemble_logits(models, images)
        probs = torch.sigmoid(logits)
        
        # 针对当前 Batch 中的每一张图片
        for b in range(len(indices)):
            s_idx = indices[b]
            c_list = sample_to_cand_classes[s_idx] # 这张图里有哪些标签是 Candidate?
            
            for (c_idx, global_i) in c_list:
                # 1. 计算置信度 (偏离 0.5 的程度)
                conf_arr[global_i] = torch.abs(probs[b, c_idx] - 0.2).detach()
                
                # 2. 核心修复：计算特定标签 Loss 对输入图片 x 的梯度范数
                loss_c = F.binary_cross_entropy_with_logits(logits[b:b+1, c_idx], targets[b:b+1, c_idx].float())
                
                # retain_graph=True 是必须的，因为同一张图片可能存在多个标签都是 Candidate
                g = torch.autograd.grad(loss_c, images, retain_graph=True)[0]
                
                # 提取对应图片的梯度并求范数
                g_img = g[b]
                gnorm = torch.norm(g_img.flatten())
                grad_arr[global_i] = gnorm.detach()

        # 释放梯度内存
        images.requires_grad_(False)

    # ==========================================================
    # 阶段 3：执行 MEE Refinement (高 Conf AND 低 Grad)
    # ==========================================================
    conf_thresh = torch.quantile(conf_arr, 1.0 - args.top_conf_ratio)
    grad_thresh = torch.quantile(grad_arr, args.low_grad_ratio)
    
    is_mee = (conf_arr >= conf_thresh) & (grad_arr <= grad_thresh)
    mee_local_idx = torch.arange(num_candidates)[is_mee.cpu()]
    
    mee_count = len(mee_local_idx)
    logger.info(f"    -> MEE Refinement Thresholds: Conf >= {conf_thresh:.4f}, Grad <= {grad_thresh:.4f}")
    logger.info(f"    -> Final MEE Labels CUT from Clean pool: {mee_count}")
    
    mee_noisy_mask = torch.zeros_like(fkl_mask)
    if mee_count > 0:
        final_mee_indices = cand_global_indices[mee_local_idx]
        mee_noisy_mask[final_mee_indices[:, 0], final_mee_indices[:, 1]] = True
            
    return fkl_mask, mee_noisy_mask
def pick_remove_rate_by_phase(rn, i1, i2, i3, r1, r2, r3, r4):
    """根据当前的 Round 轮次，匹配对应的 remove_rate"""
    if rn <= i1:
        return r1
    if rn <= i1 + i2:
        return r2
    if rn <= i1 + i2 + i3:
        return r3
    return r4
def get_fkl_required_epochs(rn, i1, i2, i3):
    """根据 MEE 原版逻辑，通过 i_rate 确定当前所处的 Phase，返回需要连续预测正确的次数"""
    if rn <= i1:
        return 1
    elif rn <= i1 + i2:
        return 2
    elif rn <= i1 + i2 + i3:
        return 3
    else:
        return 4
@torch.inference_mode()
def validate_with_meter(models, val_loader, device):
    """
    使用 metric.py 中的 AveragePrecisionMeter 进行全面验证
    """
    for m in models: 
        m.eval()
        
    # 实例化你提供的评估器
    meter = AveragePrecisionMeter()
    
    for images, targets, _ in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # MEE 方法：获取四个网络的集成 logits
        logits = ensemble_logits(models, images)
        
        # 将当前 batch 的 logits 和真实标签输入到 meter 中
        # 注意：meter 内部源码做了 sigmoid，所以这里直接传 logits 即可
        meter.add(logits.detach(), targets.detach())
        
    # 计算并返回所有指标的字典
    metrics_dict = meter.compute_all_metrics()
    return metrics_dict
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
    # ================= 🌟 新增：定义验证集 Loader =================
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True  # 推荐加上，可以加速数据转移到显存
    )
    # ===============================================================
    criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.loss_clip, disable_torch_grad_focal_loss=False)

    # 提取全局 Target 矩阵，用于最终的 Clean > Noisy 投票
    etrain_loader = torch.utils.data.DataLoader(train_dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    num_train_samples = len(train_dataset)
    all_targets = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)
    #取出训练集中的所有标签，根据索引，保存True或False到all_targets中
    for _, targets, indices in etrain_loader:
        all_targets[indices] = (targets == 1).to(device)

    # 标签级FkL矩阵初始化
    fkl_consecutive_counts = torch.zeros((num_train_samples, args.num_class), dtype=torch.int32).to(device)
    fkl_mask = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)

    # === 初始化全局掩码 ===
    # 初始状态下，所有标签都视为“干净”（参与训练）

    global_label_mask = torch.ones((num_train_samples, args.num_class), dtype=torch.bool).to(device)
    num_rounds = args.i_rate_1 + args.i_rate_2 + args.i_rate_3 + args.i_rate_4
    logger.info(f"=================== Stage 1 Training Start (Multi-Round Mode: {num_rounds} Rounds) ===================")
    # ================= 🌟 补充缺失的初始化 =================
    best_global_auroc = 0.0
    metrics_csv_path = os.path.join(args.output, 'training_metrics.csv')
    with open(metrics_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Epoch', 'Val_mAUC', 'Val_mAcc', 'Val_Macro_F1', 'FkL_Candidates', 'Remaining_Clean_Pool'])
    # ========================================================
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    
    # ================= 修改三：真正激活 Multi-Round 逻辑 =================
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n" + "="*20 + f" Starting Round {round_num}/{num_rounds} " + "="*20)

        # ================= 关键修改：每轮重新初始化网络与优化器 =================
        net1 = build_q2l(args).to(device)
        net2 = build_q2l(args).to(device)
        ema1 = ModelEMA(net1, alpha=0.999)
        ema2 = ModelEMA(net2, alpha=0.999)
        models = (net1, net2, ema1.ema_model, ema2.ema_model)

        opt1, opt2 = build_dual_optimizers(net1, net2, args)
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs, eta_min=1e-5)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs, eta_min=1e-5)
        # ================= 关键修改：每轮重置 FkL 统计矩阵 =================
        # 因为网络重置了，连续正确预测的次数也必须重新从 0 开始计算
        fkl_consecutive_counts = torch.zeros((num_train_samples, args.num_class), dtype=torch.int32).to(device)
        fkl_mask = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)
        for epoch in range(args.epochs):
            logger.info(f"\n[Round {round_num}/{num_rounds}] Epoch {epoch + 1}/{args.epochs}")
            
            # 1. 训练一个 Epoch (现在正确传入了 global_label_mask)
            train_one_epoch_mutual_kd(net1, net2, ema1.ema_model, ema2.ema_model, ema1, ema2, opt1, opt2, train_loader, criterion, args, device, global_label_mask)
            sch1.step()
            sch2.step()
            # ================== 🌟 直接调用 Meter 验证并保存最佳权重 ==================
            val_metrics = validate_with_meter(models, val_loader, device)
            
            # 提取你关心的核心指标
            val_auroc = val_metrics.get('mAUC', 0.0)
            val_acc = val_metrics.get('mAcc', 0.0)
            val_macro_f1 = val_metrics.get('macro_F1', 0.0)
            
            logger.info(f"    -> Epoch {epoch + 1} Val mAUC: {val_auroc:.4f} | mAcc: {val_acc:.4f} | Macro F1: {val_macro_f1:.4f}")
            
            # 仍然使用 mAUC 作为保存最佳模型的唯一标准 (医学影像多标签的黄金标准)
            if val_auroc > best_global_auroc:
                best_global_auroc = val_auroc
                best_ckpt_path = os.path.join(args.output, 'best_stage1_model.pth')
                
                logger.info(f"    >>> [Best Model] Saving new best model (mAUC: {best_global_auroc:.4f}) to {best_ckpt_path} <<<")
                torch.save({
                    'round': round_num,
                    'epoch': epoch,
                    'net1': net1.state_dict(),
                    'net2': net2.state_dict(),
                    'net1_ema': ema1.ema_model.state_dict(),
                    'net2_ema': ema2.ema_model.state_dict(),
                    'best_auroc': best_global_auroc
                }, best_ckpt_path)
            # ================================================================
            # 2. 追踪标签级 FkL
            if epoch >= args.warm_up_epochs:
                logger.info("    -> Tracking LABEL-LEVEL predictions for FkL...")
                
                # ========== 【核心修正：严格对齐 MEE 的 Phase 逻辑】 ==========
                # 根据当前 round_num 和 i_rate 划分，算出真实的 FkL 连续要求次数
                current_required_epochs = get_fkl_required_epochs(round_num, args.i_rate_1, args.i_rate_2, args.i_rate_3)
                current_required_epochs = min(current_required_epochs, args.fkl_consecutive_epochs)
                
                logger.info(f"    -> Current Round {round_num} is in Phase {current_required_epochs}, requires {current_required_epochs} consecutive correct predictions.")
                
                update_fkl_mask(models, etrain_loader, device, fkl_consecutive_counts, fkl_mask, current_required_epochs)
                # ========== 【核心修正：严格基于“有标注的正标签”计算进度】 ==========
                # 你的 all_targets 矩阵完美记录了所有 Target == 1 的位置
                
                # 1. 统计当前满足连续预测正确的候选者中，属于“真正疾病(正标签)”的数量
                current_fkl_count = (fkl_mask & all_targets).sum().item()
                
                # 2. 获取本阶段的保留比例
                current_remove_rate = pick_remove_rate_by_phase(
                    round_num,
                    args.i_rate_1, args.i_rate_2, args.i_rate_3,
                    args.remove_rate_1, args.remove_rate_2, args.remove_rate_3, args.remove_rate_4
                )
                
                # 3. 获取当前池子里，依然存活的“真正疾病(正标签)”总数
                current_pool_size = (global_label_mask & all_targets).sum().item()
                
                # 4. 动态算出基于正样本的及格线
                dynamic_threshold = int(current_pool_size * current_remove_rate)
                # ====================================================================
                
                logger.info(f"    -> Current FkL Candidates: {current_fkl_count} (Dynamic Threshold: {dynamic_threshold})")
                logger.info(f"    -> Current pool size: {current_pool_size})")
                # === 核心逻辑：使用动态及格线判断是否达到截断条件 ===
                if current_fkl_count >= dynamic_threshold or epoch == args.epochs - 1:
                    # ==========================================================
                    # 🌟 核心修复：无条件信任负样本（给所有 target == 0 颁发免死金牌）
                    # ==========================================================
                    negative_targets_mask = ~all_targets
                    if round_num == 1:
                        logger.info(f"\n>>> [Round 1] Executing Label-Level Early Cutting (Blacklist Mode)... <<<")
                        fkl_mask, mee_noisy_mask = perform_label_level_early_cutting(models, train_dataset_eval, fkl_mask, args, logger, device,all_targets)
                        
                        # ⚠️ 真正的干净标签 = (FkL选出的正标签 OR 默认干净的负标签) AND (非MEE黑名单)
                        global_label_mask = global_label_mask & (fkl_mask | negative_targets_mask) & (~mee_noisy_mask)
                    else:
                        logger.info(f"\n>>> [Round {round_num}] Using pure FkL filtering (Whitelist Mode)... <<<")
                        
                        # ⚠️ 真正的干净标签 = (FkL选出的正标签 OR 默认干净的负标签)
                        global_label_mask = global_label_mask & (fkl_mask | negative_targets_mask)    
                    # 判断是否是最后一轮
                    if round_num == num_rounds:
                        logger.info(f"\n>>> Final Round {round_num} Reached. Executing Sample-Level Voting & Exiting! <<<")
                        
                        # 【阶段B】样本级聚合投票 (核心: clean_pos > noisy_pos)
                        total_positives = all_targets.sum(dim=1) 
                        clean_pos_counts = (global_label_mask & all_targets).sum(dim=1)
                        noisy_pos_counts = (~global_label_mask & all_targets).sum(dim=1)
                        
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
                            clean_lbls = torch.nonzero(global_label_mask[n_idx]).squeeze(1).tolist()
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