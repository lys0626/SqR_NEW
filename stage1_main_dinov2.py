import argparse
import os, sys
import random
import datetime
import time
import json
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
import csv
from lib.dataset.get_dataset import get_datasets
from lib.utils.logger import setup_logger
from lib.utils.metric import AveragePrecisionMeter
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict
from collections import defaultdict
from lib.models.aslloss import AsymmetricLossOptimized

# ================= 新增：DINOv2 模型包装类 =================
class DINOClassifier(nn.Module):
    def __init__(self, num_class, arch='dinov2_vits14'):
        super().__init__()
        # 1. 本地代码路径 (替换为你实际上传的 dinov2 仓库目录)
        local_repo_path = '/data/dsj/lys/dinov2' 
        # 2. 本地权重路径 (替换为你实际上传的 .pth 文件路径)
        local_weight_path = '/data/dsj/lys/dinov2_vits14_pretrain.pth'
        # ================= 核心离线加载逻辑 =================
        # source='local' 表示从本地目录读取网络架构代码
        # pretrained=False 表示只构建空骨架，绝不向外网请求权重！
        self.backbone = torch.hub.load(local_repo_path, arch, source='local', pretrained=False)
        # 手动将本地的 .pth 权重文件“塞”进空骨架里
        state_dict = torch.load(local_weight_path, map_location='cpu')
        self.backbone.load_state_dict(state_dict)
        # ====================================================
        # 获取 DINO 的特征维度 (vits14 为 384)
        embed_dim = self.backbone.embed_dim
        # 替换为多标签分类头
        self.fc = nn.Linear(embed_dim, num_class)
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
def build_dino(args):
    # 使用 vits14 版本，对显存较友好
    return DINOClassifier(num_class=args.num_class, arch='dinov2_vits14')
# ==========================================================

def sec_to_str(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def parser_args():
    parser = argparse.ArgumentParser(description='DINOv2 NIH Training (Stage 1 - MEE Label-Level)')
    # == 原有基础参数 ==
    parser.add_argument('--dataname', default='nih', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', default='/comp_robot')
    parser.add_argument('--img_size', default=224, type=int) # 注意：必须是 14 的倍数
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
    parser.add_argument('--lr', default=1e-4, type=float) # 建议：DINO 微调时 LR 调小一点
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--seed', default=95, type=int)

    # ================= 标签级 MEE 核心对齐参数 =================
    parser.add_argument('--warm_up_epochs', default=6, type=int, help='前几个 Epoch 不记录 FkL')
    parser.add_argument('--fkl_consecutive_epochs', default=5, type=int, help='需要连续多少次预测正确才算候选干净标签')
    
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
    parser.add_argument("--i_rate_3", type=int, default=0)
    parser.add_argument("--i_rate_4", type=int, default=0)
    
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
            # ======= 修改：适配 DINO 的参数命名 =======
            if 'fc' in name: 
                other.append(param)
            else: 
                backbone.append(param)
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
    # ======= 修改：去除了解包操作 =======
    out1 = net1(x)
    out2 = net2(x)
    out1e = net1e(x)
    out2e = net2e(x)
    return (out1 + out2 + out1e + out2e) / 4.0
import torchvision.transforms as T

# 在 stage1_main_dinov2.py 文件顶部引入 RandomErasing
# RandomErasing 是一种极佳的非对称扰动，能强迫模型关注不同的局部病灶区域
random_erasing = T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
def train_one_epoch_mutual_kd(net1, net2, net1e, net2e, ema1, ema2, opt1, opt2, loader, criterion, args, device, global_label_mask,sch1,sch2,current_epoch):
    net1.train()
    net2.train()
    net1e.eval()
    net2e.eval()
    temperature = 2.0
    alpha = 0.5
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for images, targets, indices in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        
        batch_mask = global_label_mask[indices].float()
        # ==========================================
        # 🌟 修改 2（终极版）：独立随机非对称增强
        # ==========================================
        # 分别对两份独立的 clone 数据进行擦除操作。
        # 因为 p=0.5，这会自动产生 4 种组合，概率各占 25%：
        # 1. net1 原图，net2 遮挡 (net2 猜 net1)
        # 2. net1 遮挡，net2 原图 (net1 猜 net2)
        # 3. net1 遮挡，net2 遮挡 (都在不同位置被遮挡，互相猜)
        # 4. net1 原图，net2 原图 (标准的原始互学习)
        
        images_net1 = random_erasing(images.clone())
        images_net2 = random_erasing(images.clone())
        with torch.cuda.amp.autocast(enabled=args.amp):
            # ======= 修改：直接接收 logits =======
            z1 = net1(images_net1)
            z2 = net2(images_net2)
            
            with torch.no_grad():
                z1e = net1e(images)
                z2e = net2e(images)

            soft1 = torch.sigmoid(z1.detach() / temperature)
            soft2 = torch.sigmoid(z2.detach() / temperature)
            soft1e = torch.sigmoid(z1e / temperature)
            soft2e = torch.sigmoid(z2e / temperature)
            
            soft1_comb = (soft1 + soft1e) / 2.0
            soft2_comb = (soft2 + soft2e) / 2.0
        
            masked_z1 = z1.clone()
            masked_z2 = z2.clone()
            
            ignore_idx = (batch_mask == 0)
            
            masked_z1[ignore_idx] = 20.0
            masked_z2[ignore_idx] = 20.0
            #ASL LOSS
            # hard1 = criterion(masked_z1, targets.float())
            # hard2 = criterion(masked_z2, targets.float())
            #DAL LOSS
            progress_ratio = current_epoch / args.epochs
            hard1 = criterion(masked_z1, targets.float(), progress_ratio)
            hard2 = criterion(masked_z2, targets.float(), progress_ratio)
            soft1_loss = F.binary_cross_entropy_with_logits(z1 / temperature, soft2_comb, reduction='none') * (temperature ** 2)
            soft2_loss = F.binary_cross_entropy_with_logits(z2 / temperature, soft1_comb, reduction='none') * (temperature ** 2)

            # loss1 = hard1 + alpha * (soft1_loss * batch_mask).sum() / max(1.0, batch_mask.sum().item())
            # loss2 = hard2 + alpha * (soft2_loss * batch_mask).sum() / max(1.0, batch_mask.sum().item())
            # 让模型在噪声位置也能向双 EMA 老师的共识靠拢，实现隐式纠错,no_mask_EMA
            loss1 = hard1 + alpha * soft1_loss.mean()
            loss2 = hard2 + alpha * soft2_loss.mean()
        opt1.zero_grad()
        scaler.scale(loss1).backward()
        scaler.unscale_(opt1)
        torch.nn.utils.clip_grad_norm_(net1.parameters(), max_norm=1.0) # 放宽了针对 Transformer 的极小裁剪阈值
        scaler.step(opt1)
        ema1.step()

        opt2.zero_grad()
        scaler.scale(loss2).backward()
        scaler.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(net2.parameters(), max_norm=1.0) 
        scaler.step(opt2)
        ema2.step()

        scaler.update()

        sch1.step()
        sch2.step()

@torch.inference_mode()
def update_fkl_mask(models, loader, device, fkl_consecutive_counts, fkl_mask, consecutive_epochs_required):
    for x, y, indices in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        
        logits = ensemble_logits(models, x)
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        is_correct_and_positive = (preds == y) & (y == 1)
        
        counts = fkl_consecutive_counts[indices]
        counts = torch.where(is_correct_and_positive, counts + 1, torch.zeros_like(counts))
        fkl_consecutive_counts[indices] = counts
        
        new_fkl = (counts >= consecutive_epochs_required)
        fkl_mask[indices] = fkl_mask[indices] | new_fkl

def perform_label_level_early_cutting(models, dataset, fkl_mask, args, logger, device,all_targets):
    for m in models: m.eval()
    
    num_samples = len(dataset)
    num_classes = args.num_class
    
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

    positive_targets = (all_targets == 1)

    fkl_indices = torch.nonzero(fkl_mask & positive_targets)
    M = len(fkl_indices) 
    logger.info(f"    -> Found {M} FkL label candidates.")

    if M == 0:
        return fkl_mask, torch.zeros_like(fkl_mask)

    fkl_losses = loss_tensor[fkl_mask]
    num_candidates = int(M / args.early_cutting_rate)
    num_candidates = min(args.newremove_rate, num_candidates)
    
    _, sorted_idx = torch.sort(fkl_losses, descending=True)
    cand_local_idx = sorted_idx[:num_candidates] 
    cand_global_indices = fkl_indices[cand_local_idx] 
    
    logger.info(f"    -> Step 2: Calculating Confidence and INPUT GRADIENT NORM for {num_candidates} specific labels...")

    sample_to_cand_classes = defaultdict(list)
    for i, (s_idx, c_idx) in enumerate(cand_global_indices.cpu().numpy()):
        sample_to_cand_classes[s_idx].append((c_idx, i))

    unique_sample_indices = list(sample_to_cand_classes.keys())
    subset = torch.utils.data.Subset(dataset, unique_sample_indices)
    
    torch.cuda.empty_cache() 
    
    grad_bs = 32  
    
    logger.info(f"    -> [Memory Protector] Re-building subset loader with ultra-small batch size: {grad_bs}")
    
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=grad_bs, shuffle=False, num_workers=args.workers)

    conf_arr = torch.zeros(num_candidates, dtype=torch.float32).to(device)
    grad_arr = torch.zeros(num_candidates, dtype=torch.float32).to(device)

    for images, targets, indices in subset_loader:
        images, targets = images.to(device), targets.to(device)
        indices = indices.numpy()
        
        images.requires_grad_(True)
        logits = ensemble_logits(models, images)
        probs = torch.sigmoid(logits)
        
        for b in range(len(indices)):
            s_idx = indices[b]
            c_list = sample_to_cand_classes[s_idx] 
            
            for (c_idx, global_i) in c_list:
                conf_arr[global_i] = torch.abs(probs[b, c_idx] - 0.5).detach()
                loss_c = F.binary_cross_entropy_with_logits(logits[b:b+1, c_idx], targets[b:b+1, c_idx].float())
                g = torch.autograd.grad(loss_c, images, retain_graph=True)[0]
                g_img = g[b]
                gnorm = torch.norm(g_img.flatten())
                grad_arr[global_i] = gnorm.detach()

        images.requires_grad_(False)

    # ================= 修改前 =================
    # conf_thresh = torch.quantile(conf_arr, 1.0 - args.top_conf_ratio)
    # grad_thresh = torch.quantile(grad_arr, args.low_grad_ratio)
    # ================= 修改后 =================
    # 1. 计算动态分位数
    dynamic_conf_thresh = torch.quantile(conf_arr, 1.0 - args.top_conf_ratio)
    dynamic_grad_thresh = torch.quantile(grad_arr, args.low_grad_ratio)
    
    # 2. 引入绝对物理意义的“保底防线”
    # 置信度 |p-0.5| 绝对不能低于 0.35 (即模型预测概率必须 >0.85 或 <0.15 才算高置信度)
    min_allowed_conf = torch.tensor(0.35).to(device)
    conf_thresh = torch.max(dynamic_conf_thresh, min_allowed_conf)
    
    # 梯度必须小于候选池平均梯度的 0.6 倍，防止 quantile 选出相对较大梯度的样本
    max_allowed_grad = grad_arr.mean() * 0.6
    grad_thresh = torch.min(dynamic_grad_thresh, max_allowed_grad)
    
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
    if rn <= i1:
        return r1
    if rn <= i1 + i2:
        return r2
    if rn <= i1 + i2 + i3:
        return r3
    return r4

def get_fkl_required_epochs(rn, i1, i2, i3):
    if rn <= i1:
        return 2
    elif rn <= i1 + i2:
        return 3
    elif rn <= i1 + i2 + i3:
        return 3
    else:
        return 4

@torch.inference_mode()
def validate_with_meter(models, val_loader, device):
    for m in models: 
        m.eval()
        
    meter = AveragePrecisionMeter()
    
    for images, targets, _ in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        logits = ensemble_logits(models, images)
        meter.add(logits.detach(), targets.detach())
        
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
    train_dataset_eval = deepcopy(train_dataset)
    train_dataset_eval.transform = val_dataset.transform
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=True  
    )
    #ASL loss
    # criterion = AsymmetricLossOptimized(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.loss_clip, disable_torch_grad_focal_loss=False)
    #D-ASL loss
    from lib.models.aslloss import DynamicAsymmetricLoss
    criterion = DynamicAsymmetricLoss(
        gamma_neg=args.gamma_neg, 
        gamma_pos=args.gamma_pos, 
        base_clip=args.loss_clip, 
        max_clip=0.2  # 您可以根据 NIH 数据集的噪声严重程度微调此值
    )
    etrain_loader = torch.utils.data.DataLoader(train_dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    num_train_samples = len(train_dataset)
    all_targets = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)
    
    for _, targets, indices in etrain_loader:
        all_targets[indices] = (targets == 1).to(device)

    global_label_mask = torch.ones((num_train_samples, args.num_class), dtype=torch.bool).to(device)
    num_rounds = args.i_rate_1 + args.i_rate_2 + args.i_rate_3 + args.i_rate_4
    logger.info(f"=================== Stage 1 Training Start (Multi-Round Mode: {num_rounds} Rounds) ===================")
    
    best_global_auroc = 0.0
    metrics_csv_path = os.path.join(args.output, 'training_metrics.csv')
    with open(metrics_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Epoch', 'Val_mAUC', 'Val_mAcc', 'Val_Macro_F1', 'FkL_Candidates', 'Remaining_Clean_Pool'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n" + "="*20 + f" Starting Round {round_num}/{num_rounds} " + "="*20)

        # ======= 修改：将构建 Q2L 替换为构建 DINO =======
        net1 = build_dino(args).to(device)
        net2 = build_dino(args).to(device)
        
        # # ==========================================
        # # 🌟 进阶修复：表征继承 + 决策头重置 (Head Reset)
        # # ==========================================
        # best_ckpt_path = os.path.join(args.output, 'best_stage1_model.pth')
        # if round_num > 1 and os.path.exists(best_ckpt_path):
        #     logger.info(f"    -> [Continual Learning] Restoring Backbone weights from Round {round_num-1}...")
        #     checkpoint = torch.load(best_ckpt_path, map_location=device)
            
        #     # 加载全部权重
        #     net1.load_state_dict(checkpoint['net1'])
        #     net2.load_state_dict(checkpoint['net2'])
            
        #     # 【核心逻辑】：重置分类器 (fc) 的权重和偏置
        #     # 使用 Xavier 均匀分布初始化权重，将偏置清零
        #     torch.nn.init.xavier_uniform_(net1.fc.weight)
        #     torch.nn.init.zeros_(net1.fc.bias)
            
        #     torch.nn.init.xavier_uniform_(net2.fc.weight)
        #     torch.nn.init.zeros_(net2.fc.bias)
            
        #     logger.info(f"    -> [Head Reset] Successfully re-initialized the Classifier Heads to clear Confirmation Bias!")

        # 2. 利用继承了 Backbone 但重置了 Head 的 net1/net2 初始化 EMA 老师
        ema1 = ModelEMA(net1, alpha=0.999)
        ema2 = ModelEMA(net2, alpha=0.999)
        models = (net1, net2, ema1.ema_model, ema2.ema_model)

        opt1, opt2 = build_dual_optimizers(net1, net2, args)
        
        found_lrs1 = [group['lr'] for group in opt1.param_groups]
        found_lrs2 = [group['lr'] for group in opt2.param_groups]

        sch1 = torch.optim.lr_scheduler.OneCycleLR(
            opt1, 
            max_lr=found_lrs1, 
            steps_per_epoch=len(train_loader), 
            epochs=args.epochs, 
            pct_start=0.2 
        )
        sch2 = torch.optim.lr_scheduler.OneCycleLR(
            opt2, 
            max_lr=found_lrs2, 
            steps_per_epoch=len(train_loader), 
            epochs=args.epochs, 
            pct_start=0.2
        )
        
        fkl_consecutive_counts = torch.zeros((num_train_samples, args.num_class), dtype=torch.int32).to(device)
        fkl_mask = torch.zeros((num_train_samples, args.num_class), dtype=torch.bool).to(device)
        
        for epoch in range(args.epochs):
            logger.info(f"\n[Round {round_num}/{num_rounds}] Epoch {epoch + 1}/{args.epochs}")
            
            train_one_epoch_mutual_kd(net1, net2, ema1.ema_model, ema2.ema_model, ema1, ema2, opt1, opt2, train_loader, criterion, args, device, global_label_mask,sch1,sch2,epoch)
            
            val_metrics = validate_with_meter(models, val_loader, device)
            
            val_auroc = val_metrics.get('mAUC', 0.0)
            val_macro_f1 = val_metrics.get('macro_F1', 0.0)
            
            logger.info(f"    -> Epoch {epoch + 1} Val mAUC: {val_auroc:.4f} | Macro F1: {val_macro_f1:.4f}")
            
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
            
            if epoch >= args.warm_up_epochs:
                logger.info("    -> Tracking LABEL-LEVEL predictions for FkL...")
                
                current_required_epochs = get_fkl_required_epochs(round_num, args.i_rate_1, args.i_rate_2, args.i_rate_3)
                current_required_epochs = min(current_required_epochs, args.fkl_consecutive_epochs)
                
                logger.info(f"    -> Current Round {round_num} is in Phase {current_required_epochs}, requires {current_required_epochs} consecutive correct predictions.")
                
                update_fkl_mask(models, etrain_loader, device, fkl_consecutive_counts, fkl_mask, current_required_epochs)
                
                current_fkl_count = (fkl_mask & global_label_mask & all_targets).sum().item()
                current_remove_rate = pick_remove_rate_by_phase(
                    round_num,
                    args.i_rate_1, args.i_rate_2, args.i_rate_3,
                    args.remove_rate_1, args.remove_rate_2, args.remove_rate_3, args.remove_rate_4
                )
                
                current_pool_size = (global_label_mask & all_targets).sum().item()
                dynamic_threshold = int(current_pool_size * current_remove_rate)
                
                logger.info(f"    -> Current FkL Candidates: {current_fkl_count} (Dynamic Threshold: {dynamic_threshold})")
                logger.info(f"    -> Current pool size: {current_pool_size})")
                
                if current_fkl_count >= dynamic_threshold or epoch == args.epochs - 1:
                    negative_targets_mask = ~all_targets
                    if round_num == 1:
                        logger.info(f"\n>>> [Round 1] Executing Label-Level Early Cutting (Blacklist Mode)... <<<")
                        fkl_mask, mee_noisy_mask = perform_label_level_early_cutting(models, train_dataset_eval, fkl_mask, args, logger, device,all_targets)
                        
                        global_label_mask = global_label_mask & (fkl_mask | negative_targets_mask) & (~mee_noisy_mask)
                    else:
                        logger.info(f"\n>>> [Round {round_num}] Using pure FkL filtering (Whitelist Mode)... <<<")
                        
                        global_label_mask = global_label_mask & (fkl_mask | negative_targets_mask)    
                    # 判断是否是最后一轮
                    #用于将噪声标签进行修改
                    if round_num == num_rounds:
                        logger.info(f"\n>>> Final Round {round_num} Reached. Generating Asymmetric Soft Targets (FP & FN) & Exiting! <<<")
                        
                        # ==========================================
                        # 🌟 关键修复：加载历史最佳权重，防止训练末期性能衰退
                        # ==========================================
                        best_ckpt_path = os.path.join(args.output, 'best_stage1_model.pth')
                        if os.path.exists(best_ckpt_path):
                            logger.info(f"    -> Loading BEST EMA models from {best_ckpt_path} for Soft Label generation...")
                            checkpoint = torch.load(best_ckpt_path, map_location=device)
                            # 将巅峰状态的权重强行覆盖回当前的 EMA 模型
                            ema1.ema_model.load_state_dict(checkpoint['net1_ema'])
                            ema2.ema_model.load_state_dict(checkpoint['net2_ema'])
                            logger.info(f"    -> Successfully restored EMA models to their peak performance (Val mAUC: {checkpoint['best_auroc']:.4f} at Round {checkpoint['round']}, Epoch {checkpoint['epoch']})")
                        else:
                            logger.warning("    -> Best checkpoint not found! Using the final epoch's degraded models instead.")

                        # ==========================================
                        # 0. 准备 ema_preds 和 ema_vars (新增方差矩阵)
                        # ==========================================
                        ema_preds = torch.zeros((num_train_samples, args.num_class), dtype=torch.float32).to(device)
                        ema_vars = torch.zeros((num_train_samples, args.num_class), dtype=torch.float32).to(device) # 追踪不确定性
                        
                        ema1.ema_model.eval()
                        ema2.ema_model.eval()
                        
                        logger.info("    -> Inferencing whole dataset with Peak-Performance Dual-EMA models...")
                        with torch.no_grad():
                            for images, _, indices in etrain_loader:
                                images = images.to(device, non_blocking=True)
                                indices = indices.to(device, non_blocking=True)
                                
                                # TTA 1: 原始图像
                                p1 = torch.sigmoid((ema1.ema_model(images) + ema2.ema_model(images)) / 2.0)
                                
                                # TTA 2: 水平翻转图像
                                images_hflip = torch.flip(images, dims=[3])
                                p2 = torch.sigmoid((ema1.ema_model(images_hflip) + ema2.ema_model(images_hflip)) / 2.0)
                                
                                # 计算均值和方差
                                mean_probs = (p1 + p2) / 2.0
                                var_probs = torch.var(torch.stack([p1, p2]), dim=0)
                                
                                ema_preds[indices] = mean_probs
                                ema_vars[indices] = var_probs

                        # ==========================================
                        # 🌟 修改 4 & 5 融合：带共现约束与不确定性过滤的 FN 挖掘
                        # ==========================================
                        final_soft_targets = all_targets.float().clone()

                        # 处理 FP (保持不变，但加入极值压制防坍塌)
                        fp_mask = (all_targets == 1) & (~global_label_mask)
                        # final_soft_targets[fp_mask] = torch.clamp(ema_preds[fp_mask], max=0.5) 
                        final_soft_targets[fp_mask] = torch.clamp(ema_preds[fp_mask], max=1) 

                        # ==========================================
                        # 🌟 重大升级：引入医学大模型 (LLM) 先验知识矩阵
                        # ==========================================
                        # 1. 计算当前数据集的后验统计概率 (Data-driven，含有一定的噪声偏差)
                        clean_pool_labels = (all_targets & global_label_mask).float() 
                        co_counts = torch.matmul(clean_pool_labels.T, clean_pool_labels)
                        class_counts = clean_pool_labels.sum(dim=0).clamp(min=1e-5)      
                        data_cond_matrix = co_counts / class_counts.unsqueeze(1)         

                        # 2. 根据 dataname 动态加载离线生成的 LLM 医学病理先验矩阵
                        # 文件名格式: medical_prior_matrix_{dataname}.npy
                        prior_filename = f'medical_prior_matrix_{args.dataname}.npy'
                        prior_path = os.path.join('/data/dsj/lys/SqR-NEW/', prior_filename)
                        try:
                            # 尝试以二进制模式加载 numpy 数组
                            matrix_np = np.load(prior_path, allow_pickle=True).astype(np.float32)
                            llm_prior_matrix = torch.from_numpy(matrix_np).to(device)
                            
                            # 检查加载的矩阵维度是否与当前任务的类别数匹配
                            if llm_prior_matrix.shape[0] != args.num_class:
                                raise ValueError(f"Matrix shape {llm_prior_matrix.shape} does not match args.num_class {args.num_class}")
                                
                            logger.info(f"    -> Successfully loaded LLM Medical Prior Matrix: {prior_filename}")
                            
                        except Exception as e:
                            # 如果文件不存在、损坏或维度不匹配，发出警告并回退到纯数据驱动矩阵
                            logger.warning(f"    -> [WARNING] Failed to load LLM matrix {prior_filename}. Reason: {e}")
                            logger.warning("    -> Falling back to purely data-driven matrix to prevent crash.")
                            llm_prior_matrix = data_cond_matrix 

                        # 3. 混合先验 (Hybrid Prior): 结合病理常识与当前医院数据的真实分布
                        alpha = 0.7  # 信任 LLM 的权重 (0.7 表示以医学常识为主，数据统计为辅)
                        cond_prob_matrix = alpha * llm_prior_matrix + (1.0 - alpha) * data_cond_matrix       

                        fn_mask = torch.zeros_like(all_targets, dtype=torch.bool).to(device)
                        
                        # 2. 逐类别进行受控的 FN 挖掘
                        for c in range(args.num_class):
                            # A. TTA 均值高，且方差小 (模型认知极其坚定)
                            high_conf_mask = (ema_preds[:, c] > 0.7) & (ema_vars[:, c] < 0.2)
                            # 候选的漏诊样本
                            candidate_fn = (all_targets[:, c] == 0) & high_conf_mask
                            
                            if not candidate_fn.any():
                                continue
                                
                            # B. 共现先验过滤 (Contextual Validation)
                            # 如果疾病 c 通常伴随其他疾病出现（例如平均条件概率 > 0.3），
                            # 我们需要检查当前候选样本是否具备这些“伴随疾病”的医学环境。
                            support_scores = torch.zeros(num_train_samples).to(device)
                            # 提取出对疾病 c 有较强指示意义的伴随疾病集合 (阈值可调，如 0.2)
                            strongly_correlated_classes = torch.where(cond_prob_matrix[:, c] > 0.2)[0]
                            if len(strongly_correlated_classes) > 1: # 排除自身的对角线
                                # 计算候选样本在这些关联疾病上的当前已知标签得分
                                for correlated_c in strongly_correlated_classes:
                                    if correlated_c != c:
                                        # 如果关联疾病存在，加分
                                        support_scores += all_targets[:, correlated_c].float()
                                        
                                # 只有当候选样本具有至少 1 个相关并发症环境，或者疾病 c 本身就高度独立时，才批准翻转
                                approved_fn = candidate_fn & (support_scores > 0)
                            else:
                                # 疾病 c 本身就是独立疾病，无需伴随症支持
                                approved_fn = candidate_fn



                            #上面注释解开，下面这一行就要注释掉，因为下面这一行是没有共现约束的纯数据驱动版本    
                            # approved_fn = candidate_fn  
                            fn_mask[:, c] = approved_fn

                        # 将筛选后、有医学常理支持、且模型认知坚定的预测写入软标签
                        final_soft_targets[fn_mask] = ema_preds[fn_mask]

                        fp_count = fp_mask.sum().item()
                        fn_count = fn_mask.sum().item()

                        # ==========================================
                        # 保存，传给 Stage 2 训练
                        # ==========================================
                        output_path = os.path.join(args.output, 'asymmetric_soft_targets.pt')
                        torch.save(final_soft_targets, output_path)
                        
                        logger.info("="*60)
                        logger.info(f" >>> Stage 1 Split Complete (Peak-Performance Asymmetric Soft Correction)! ")
                        logger.info(f"     * False Positives (FP) Softened: {fp_count}")
                        logger.info(f"     * False Negatives (FN) Mined:    {fn_count}")
                        logger.info(f"     * Saved soft targets to: {output_path}")
                        logger.info(" >>> Exiting Stage 1 gracefully to start Stage 2. <<<")
                        logger.info("="*60)
                        
                        sys.stdout.flush() 
                        sys.exit(0)
                    # 只处理噪声样本和干净样本的划分，不修改标签
                    # if round_num == num_rounds:
                    #     logger.info(f"\n>>> Final Round {round_num} Reached. Executing Sample-Level Voting & Exiting! <<<")
                        
                    #     total_positives = all_targets.sum(dim=1) 
                    #     clean_pos_counts = (global_label_mask & all_targets).sum(dim=1)
                    #     noisy_pos_counts = (~global_label_mask & all_targets).sum(dim=1)
                        
                    #     is_clean_sample = torch.zeros(num_train_samples, dtype=torch.bool).to(device)
                        
                    #     has_pos = total_positives > 0
                    #     is_clean_sample[has_pos] = clean_pos_counts[has_pos] > noisy_pos_counts[has_pos]
                        
                    #     no_pos = total_positives == 0
                    #     is_clean_sample[no_pos] = True
                        
                    #     clean_indices = torch.nonzero(is_clean_sample).squeeze(1).tolist()
                    #     noisy_indices = torch.nonzero(~is_clean_sample).squeeze(1).tolist()
                        
                    #     noise_clean_labels_dict = {}
                    #     for n_idx in noisy_indices:
                    #         clean_lbls = torch.nonzero(global_label_mask[n_idx]).squeeze(1).tolist()
                    #         noise_clean_labels_dict[int(n_idx)] = clean_lbls

                    #     torch.save(clean_indices, os.path.join(args.output, 'clean_indices.pt'))
                    #     torch.save(noisy_indices, os.path.join(args.output, 'noisy_indices.pt'))
                    #     torch.save(noise_clean_labels_dict, os.path.join(args.output, 'noise_clean_labels_dict.pt'))
                        
                    #     logger.info("="*60)
                    #     logger.info(f" >>> Stage 1 Split Complete! ")
                    #     logger.info(f"     * Clean Samples: {len(clean_indices)}")
                    #     logger.info(f"     * Noisy Samples: {len(noisy_indices)}")
                    #     logger.info(" >>> Exiting Stage 1 gracefully to start Stage 2  . <<<")
                    #     logger.info("="*60)
                        
                    #     sys.stdout.flush() 
                    #     sys.exit(0)
                    else:
                        logger.info(f">>> Round {round_num} Completed. Breaking inner loop to start Round {round_num + 1}. <<<")
                        break  

if __name__ == '__main__':
    main()