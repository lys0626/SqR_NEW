import argparse
import os, sys
import random
import datetime
import time
import json
import numpy as np
from copy import deepcopy
import torch
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

# 引入 RoLT 模块
from rolt_handler import RoLT_Handler

def sec_to_str(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"

def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label NIH Training (Stage 1 - Data Cleaning)')
    parser.add_argument('--dataname', help='dataname', default='nih', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot')
    parser.add_argument('--img_size', default=224, type=int, help='size of input images')

    parser.add_argument('--output', metavar='DIR', help='path to output folder')
    parser.add_argument('--num_class', default=14, type=int, help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model.')
    parser.add_argument('--optim', default='SGD', type=str, choices=['AdamW', 'Adam_twd', 'SGD'], help='which optim to use')
    
    parser.add_argument('--scheduler', default='OneCycle', type=str, choices=['OneCycle', 'StepLR'], help='Which scheduler to use')
    parser.add_argument('--step_size', default=40, type=int, help='Period of learning rate decay (epochs) for StepLR')
    parser.add_argument('--gamma', default=0.1, type=float, help='Multiplicative factor of learning rate decay for StepLR')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--val_interval', default=1, type=int, metavar='N', help='interval of validation')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M', help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M', help='start ema epoch')
    parser.add_argument('--seed', default=95, type=int, help='seed for initializing training. ')

    # Transformer params
    parser.add_argument('--enc_layers', default=1, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'))
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true')
    parser.add_argument('--keep_input_proj', action='store_true')

    parser.add_argument('--amp', action='store_true', default=False, help='apply amp')
    
    # 核心阶段控制参数
    parser.add_argument('--splicemix_start_epoch', default=10, type=int, help='Epoch to save indices and EXIT Stage 1.')
    parser.add_argument('--rolt_start_epoch', default=3, type=int, help='Epoch to start applying RoLT noise filtering.')
    
    # 单卡设备指定
    parser.add_argument('-cd', '--cuda_devices', default=[0], nargs='+', type=int, help="Cuda device ids for running")
    # 【在这里插入以下 4 行缺失的参数】
    parser.add_argument('--orid_norm', action='store_true', default=False, help='Use [0,1] normalization')
    parser.add_argument('--cutout', action='store_true', default=False, help='Use cutout')
    parser.add_argument('--n_holes', type=int, default=1, help='number of holes for cutout')
    parser.add_argument('--length', type=int, default=16, help='length of holes for cutout')
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    
    # --- 强制单卡模式配置 ---
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    gpu_id = args.cuda_devices[0] if hasattr(args, 'cuda_devices') else 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0) 
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="Stage1-Q2L")
    logger.info("Command: "+' '.join(sys.argv))
    logger.info(f"Running in SINGLE GPU Mode on Device ID: {gpu_id}")
    
    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)

    return main_worker(args, logger)

def main_worker(args, logger):
    best_mAUC = 0

    # 1. 构建模型 (无 DDP)
    model = build_q2l(args)
    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay) 

    criterion = torch.nn.BCEWithLogitsLoss()

    # 2. 优化器分组 (去掉 .module)
    base_lr = args.lr
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name or 'fc_splicemix' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
            
    logger.info(f"Optimizer Grouping: {len(backbone_params)} backbone params (lr1), {len(other_params)} other params (lr*0.1)")

    param_dicts = [
        {"params": backbone_params, "lr": base_lr*0.1},
        {"params": other_params, "lr": base_lr},
    ]

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(param_dicts, lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optim == 'Adam_twd':
        logger.warning("Adam_twd does not support backbone lr split currently.")
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(parameters, base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)          
    else:
        raise NotImplementedError

    summary_writer = SummaryWriter(log_dir=args.output)

    # 3. 加载 Checkpoint (单卡直接 map 到 cuda)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda')
            state_dict = clean_state_dict(checkpoint.get('state_dict', checkpoint.get('model')))
            
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
                
            model.load_state_dict(state_dict, strict=False) # 去掉 .module
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint.get('epoch', 'N/A')))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # 4. Data Loading (普通 DataLoader，无 DistributedSampler)
    train_dataset, val_dataset = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        metrics_res, _ = validate(val_loader, model, criterion, args, logger)
        logger.info(' * mAUC {mAUC:.5f}'.format(mAUC=metrics_res['mAUC']))
        return

    # 初始化 RoLT
    rolt_handler = RoLT_Handler(args, model, train_loader, args.num_class, args.hidden_dim)

    # 5. 学习率调度器
    found_lrs = [group['lr'] for group in optimizer.param_groups]
    if args.scheduler == 'OneCycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=found_lrs, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.5)
        args.step_per_batch = True
    elif args.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        args.step_per_batch = False
    else:
        raise NotImplementedError

    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAUCs = AverageMeter('mAUC', ':5.5f', val_only=True)
    
    clean_mask_dict = {}
    soft_label_dict = {}

    # ================= 6. 训练循环 (到达分离点即退出) =================
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        
        need_run_rolt = (epoch <= args.splicemix_start_epoch) or (not clean_mask_dict)
        if need_run_rolt:
            clean_mask_dict, soft_label_dict = rolt_handler.step(epoch)
            
        # [核心] 提取完干净索引，直接保存并退出整个 Stage 1 脚本
        if epoch == args.splicemix_start_epoch:
            logger.info(f" >>> [Stage 1 Complete] Epoch {epoch} reached. Saving clean indices... <<<")
            clean_indices = [idx for idx, is_clean in clean_mask_dict.items() if is_clean]
            save_path = os.path.join(args.output, 'clean_indices.pt')
            torch.save(clean_indices, save_path)
            logger.info(f" >>> Saved {len(clean_indices)} clean samples to {save_path}. Exiting Stage 1 gracefully! <<<")
            # 【修复代码】防止 PyTorch 多进程 DataLoader 在清理时引发非 0 退出码
            import sys
            sys.stdout.flush() # 强制刷新控制台输出缓冲区，防止日志被截断
            sys.exit(0)        # 强行以 0 (成功) 状态码退出，顺利向外层的 bash set -e 交差
            
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        n_total = len(train_dataset)
        n_clean = n_total if len(clean_mask_dict) == 0 else (n_total - sum(1 for v in clean_mask_dict.values() if not v))

        train_start = time.time()
        
        # 只传入必需的参数
        loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, clean_mask_dict, soft_label_dict)
        
        train_duration = sec_to_str(time.time() - train_start)

        if not getattr(args, 'step_per_batch', True):
            scheduler.step()

        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = " ".join([f"{lr:.5g}" for lr in current_lrs])
        
        logger.info(f"[Epoch {epoch}, lr[{lr_str} ]] [Train] time:{train_duration}s, loss: {loss:.4f} .")
        
        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', current_lrs[0], epoch)

        # 验证阶段
        if epoch % args.val_interval == 0:
            val_start = time.time()
            metrics_res, val_loss = validate(val_loader, model, criterion, args, logger)
            metrics_res_ema, val_loss_ema = validate(val_loader, ema_m.module, criterion, args, logger)
            val_duration = sec_to_str(time.time() - val_start)

            logger.info(f" -> Main Model mAUC: {metrics_res['mAUC']:.4f}")
            logger.info(f" -> EMA Model mAUC:  {metrics_res_ema['mAUC']:.4f}")
            
            mAUC = metrics_res['mAUC']
            mi_f1, ma_f1 = metrics_res['micro_F1'], metrics_res['macro_F1']
            mi_p, ma_p = metrics_res['micro_P'], metrics_res['macro_P']
            mi_r, ma_r = metrics_res['micro_R'], metrics_res['macro_R']

            losses.update(val_loss)
            mAUCs.update(mAUC)

            # Stage 1 都是预热，不评选 best
            is_best = False

            timestamp = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            log_prefix = f"[0|{timestamp}]"
            logger.info(f"{log_prefix}: [Test] time: {val_duration}s, loss: {val_loss:.4f}, mAUC: {mAUC:.4f}, miF1: {mi_f1:.4f}, maF1: {ma_f1:.4f}")

            if summary_writer:
                summary_writer.add_scalar('val_mAUC', mAUC, epoch)
                summary_writer.add_scalar('val_loss', val_loss, epoch)

            log_txt_path = os.path.join(args.output, 'log.txt')
            with open(log_txt_path, 'a') as f:
                header = "\nEpoch\tTrain_Loss\tVal_Loss\tVal_mAUC\tmi_F1\tma_F1\tmi_R\tma_R\tmi_P\tma_P\tClean_S\tTotal_S\n"
                f.write(header)
                log_line = f"{epoch}\t{loss:.5f}\t{val_loss:.5f}\t{mAUC:.5f}\t{mi_f1:.5f}\t{ma_f1:.5f}\t{mi_r:.5f}\t{ma_r:.5f}\t{mi_p:.5f}\t{ma_p:.5f}\t{n_clean}\t{n_total}\n"
                f.write(log_line)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.backbone,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, clean_mask_dict, soft_label_dict):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses = AverageMeter('Loss', ':5.3f')
    model.train()
    
    for i, (images, target, indices) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)

        if len(clean_mask_dict) == 0:
            batch_clean_mask = torch.ones(images.size(0), dtype=torch.bool).cuda()
        else:
            batch_clean_mask = torch.tensor([clean_mask_dict.get(idx.item(), True) for idx in indices]).bool().cuda()
        
        batch_noisy_mask = ~batch_clean_mask
        clean_idxs = torch.where(batch_clean_mask)[0]
        noisy_idxs = torch.where(batch_noisy_mask)[0]

        loss_q2l_branch = 0.0
        with torch.cuda.amp.autocast(enabled=args.amp):
            out_trans_all, _, _, _ = model(images)
            
            if epoch < args.rolt_start_epoch:
                loss_q2l_branch = criterion(out_trans_all, target)
            else :
                valid_parts = 0
                if len(clean_idxs) > 0:
                    loss_clean = criterion(out_trans_all[clean_idxs], target[clean_idxs])
                    loss_q2l_branch += loss_clean
                    valid_parts += 1
                
                if len(noisy_idxs) > 0:
                    soft_targets_list = []
                    for k in noisy_idxs:
                        global_idx = indices[k].item()
                        s_label = soft_label_dict.get(global_idx, target[k])
                        soft_targets_list.append(s_label)
                    
                    if len(soft_targets_list) > 0:
                        soft_targets = torch.stack(soft_targets_list).to(images.device)
                        loss_noisy = criterion(out_trans_all[noisy_idxs], soft_targets)
                        loss_q2l_branch += loss_noisy
                        valid_parts += 1
                
                if valid_parts > 0:
                    loss_q2l_branch /= valid_parts
        
        final_loss = loss_q2l_branch

        if isinstance(final_loss, torch.Tensor) and final_loss.item() > 0:
            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            scaler.step(optimizer)
            scaler.update()
            
            if getattr(args, 'step_per_batch', False):
                scheduler.step()

        if epoch >= args.ema_epoch:
            ema_m.update(model)

        losses.update(final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss, images.size(0))

    return losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, args, logger): 
    meter = AveragePrecisionMeter(difficult_examples=False)
    losses = AverageMeter('Loss', ':5.3f') 
    model.eval()
    
    for i, (images, target, _) in enumerate(val_loader): 
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            out_logits, _, _, _ = model(images)
            output = out_logits       
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            meter.add(output, target, filename=[]) 

    metrics_res = meter.compute_all_metrics()
    return metrics_res, losses.avg

# --- 工具类保持不变 ---
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        best_path = os.path.join(os.path.split(filename)[0], 'model_best.pth.tar')
        torch.save(state, best_path)

class AverageMeter(object):
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name; self.fmt = fmt; self.val_only = val_only; self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count
    def __str__(self):
        if self.val_only: return ('{name} {val' + self.fmt + '}').format(**self.__dict__)
        else: return ('{name} {val' + self.fmt + '} ({avg' + self.fmt + '})').format(**self.__dict__)

if __name__ == '__main__':
    main()