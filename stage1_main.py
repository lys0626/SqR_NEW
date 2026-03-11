import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy
import distutils.version
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter

from lib.dataset.get_dataset import get_datasets

from lib.utils.logger import setup_logger
import lib.models as models
import lib.models.aslloss
from lib.models.query2label import build_q2l
# 修改引用路径：使用新的 Metric 工具
from lib.utils.metric import AveragePrecisionMeter
from lib.utils.misc import clean_state_dict
from lib.utils.slconfig import get_raw_dict

# --- [新增] 引入自定义模块 ---
# 假设这些文件都在根目录下
from rolt_handler import RoLT_Handler
from SpliceMix import SpliceMix
import torch.nn.functional as F
class Splicemix_CL_Loss_fn(torch.nn.Module):
    """提取自 resnet50 仓库的 Loss_fn"""
    def __init__(self):
        super(Splicemix_CL_Loss_fn, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        if isinstance(inputs, tuple) and len(inputs) == 3:
            preds, preds_m, preds_m_r = inputs
            loss_bce = self.bce(preds, targets)
            loss_cl = self.bce(preds_m, preds_m_r.sigmoid().detach())
            loss = loss_bce + loss_cl
        else:
            loss = self.bce(inputs, targets)
        return loss

def get_splicemix_outputs(model, feas, preds, flag):
    """
    提取自 resnet50 仓库，将特征切分逻辑放在外部执行，不修改模型 forward
    """
    mix_ind, mix_dict = flag['mix_ind'], flag['mix_dict']
    feas_r, preds_r = feas[(1 - mix_ind).bool()], preds[(1 - mix_ind).bool()]
    feas_m = feas[mix_ind.bool()]
    bs_m, C, h, w = feas_m.shape

    ng_list = []
    preds_m = torch.tensor([], device=feas.device)
    preds_m_r = torch.tensor([], device=feas.device)
    
    glb_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))

    for i, (rand_ind, g_row, g_col, n_drop, drop_ind) in enumerate(zip(
        mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], 
        mix_dict['n_drops'], mix_dict['drop_inds'])):
        
        ng = len(rand_ind) // (g_row * g_col)
        fea_m = feas_m[sum(ng_list): sum(ng_list) + ng]
        ng_list.append(ng)
        
        if h % g_row + w % g_col != 0:
            fea_m = F.interpolate(fea_m, (h // g_row * g_row, w // g_col * g_col), mode='bilinear', align_corners=True)
        
        chunks = [c.split(fea_m.shape[-1] // g_col, dim=-1) for c in fea_m.split(fea_m.shape[-2] // g_row, dim=-2)]
        fea_m = torch.stack([torch.stack(c, dim=1) for c in chunks], dim=1)
        fea_m = fea_m.view(-1, C, fea_m.shape[-2], fea_m.shape[-1])

        pred_m_r = torch.masked_fill(preds_r[rand_ind], drop_ind[:, None]==1, -1e3)

        fea_m_gp = glb_pooling(fea_m).flatten(1)
        
        # 调用分类头预测切块特征
        fc_head = model.module.fc_splicemix if hasattr(model, 'module') else model.fc_splicemix
        pred_m = fc_head(fea_m_gp)
        pred_m = torch.masked_fill(pred_m, drop_ind[:, None]==1, -1e3)

        preds_m = torch.cat((preds_m, pred_m), dim=0)
        preds_m_r = torch.cat((preds_m_r, pred_m_r), dim=0)

    return (preds, preds_m, preds_m_r)
# --- [新增] 时间格式化辅助函数 ---
def sec_to_str(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label MSCOCO Training')
    parser.add_argument('--dataname', help='dataname', default='nih', choices=['coco14', 'mimic', 'nih'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')

    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='SGD', type=str, choices=['AdamW', 'Adam_twd', 'SGD'],
                        help='which optim to use')
    # --- [新增] 学习率调度器参数 ---
    parser.add_argument('--scheduler', default='OneCycle', type=str, choices=['OneCycle', 'StepLR'],
                        help='Which scheduler to use: OneCycle (default) or StepLR')
    parser.add_argument('--step_size', default=40, type=int,
                        help='Period of learning rate decay (epochs) for StepLR')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Multiplicative factor of learning rate decay for StepLR')
    # -----------------------------
    # --- [新增] Momentum 参数 (SGD用) ---
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # ASL loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')  

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=95, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')


    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * training
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')

    # --- [新增] ResNet/SpliceMix 相关参数 ---
    parser.add_argument('--enable_splicemix', action='store_true', default=True,
                        help='Whether to enable SpliceMix augmentation')
    parser.add_argument('--splicemix_prob', default=1, type=float,
                        help='Probability of applying SpliceMix')
    parser.add_argument('--splicemix_mode', default='SpliceMix', type=str,
                        choices=['SpliceMix', 'SpliceMix-CL'],
                        help='Mode of SpliceMix: Standard SpliceMix or SpliceMix-CL settings')
    parser.add_argument('--splicemix_start_epoch', default=10, type=int,
                        help='Epoch to switch from Q2L warmup to SpliceMix-only training.')
    # [新增] Stage 2 专用学习率参数
    parser.add_argument('--lr_stage2', default=1e-4, type=float,
                        help='Learning rate for Stage 2 (SpliceMix training)')
    parser.add_argument('--optim_stage2', default='AdamW', type=str, choices=['AdamW', 'Adam_twd', 'SGD'],
                        help='Optimizer choice for Stage 2 (default: SGD)')
    # parser.add_argument('--stage2_backbone_init', default='reset', type=str, choices=['continue', 'reset'],
    #                     help='Stage 2 Backbone Init: "continue" (keep Stage 1 weights) or "reset" (reload ImageNet weights)')
    parser.add_argument('--scheduler_stage2', default=None, type=str, choices=['OneCycle', 'StepLR'],
                        help='Scheduler choice for Stage 2. If None, keep using --scheduler')
    parser.add_argument('--rolt_start_epoch', default=3, type=int,
                        help='Epoch to start applying RoLT noise filtering. Before this, all data is treated as clean.')
    parser.add_argument('--warmup_epochs_stage2', default=5, type=int,
                        help='Stage 2 的预热轮数 (0 表示不使用)')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args

best_mAUC = 0  # 修改变量名

def main():
    args = get_args()
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    #自动寻找随机算法，加快了训练，但是丧失了随机性
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):

    global best_mAUC

    # build model
    model = build_q2l(args)
    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay) # 0.9997 实现模型参数的指数平均移动，不参与反向传播也不会被优化器直接更新，
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                       device_ids=[args.local_rank],
                                                         broadcast_buffers=False,
                                                         find_unused_parameters=True  # <--- 必须加，单卡也得加
                                                         )

    # 使用 BCEWithLogitsLoss 替代 ASL
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer
    # --- [修改] Optimizer: 支持参数分组和 SGD ---
    args.lr_mult = args.batch_size / 256
    # base_lr = args.lr_mult * args.lr
    base_lr = args.lr
    # 1. 参数分组：Backbone与Splicemix分支的分类器 学习率x1，其他部分x0.1
    backbone_params = []
    other_params = []
    for name, param in model.module.named_parameters():
        if not param.requires_grad:
            continue
        # 根据 Query2Label 的定义，Backbone 属性名为 'backbone'
        if 'backbone' in name or 'fc_splicemix' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    # 打印分组信息确认
    if args.rank == 0:
        logger.info(f"Optimizer Grouping: {len(backbone_params)} backbone params (lr1), {len(other_params)} other params (lr*0.1)")

    param_dicts = [
        {
            "params": backbone_params, 
            "lr": base_lr    # Backbone与fc_Splicemix使用的学习率
        },
        {
            "params": other_params, 
            "lr": base_lr     # 其他部分 (Transformer, FC, Heads) 使用基础学习率
        },
    ]
    #初始化优化器
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=base_lr, # 这里的 lr 仅作为 fallback，实际生效的是 param_dicts 里的
            betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
        )
    elif args.optim == 'SGD': # [新增] SGD 逻辑
        optimizer = torch.optim.SGD(
            param_dicts,
            lr=base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    elif args.optim == 'Adam_twd':
        # Adam_twd 逻辑保持原样，如果需要分层学习率，这里的 parameters 需要重新构建
        # 鉴于 add_weight_decay 比较特殊，这里暂时维持原样
        logger.warning("Adam_twd does not support backbone lr split currently in this script.")
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            base_lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )          
    else:
        raise NotImplementedError

    # tensorboard
    #控制日志记录的权限
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None
    '''
        #args.resume参数 指向权重文件
        训练时：用于在一个较好的基础上开始训练（加载权重），省去从零收敛的时间。
        测试时：必须使用，用于加载训练好的模型进行评估。
    '''
    # ================== [新增：评估 Stage 2 模型时的结构替换] ==================
    # 如果是单独的测试模式，且测试的是 Stage 2 权重，必须提前把 Backbone 换成标准的 7x7 结构
    if args.evaluate:
        import torchvision
        if args.rank == 0:
            logger.info(" >>> [Evaluate] Swapping Backbone to Standard ResNet50 for Stage 2 testing! <<<")
        
        standard_resnet = torchvision.models.resnet50(pretrained=False) # 测试时不需要预训练权重，直接拿空壳
        standard_backbone = torch.nn.Sequential(*list(standard_resnet.children())[:-2])
        
        # 将新 backbone 放到正确的设备上
        device = next(model.parameters()).device
        standard_backbone = standard_backbone.to(device)
        
        # 替换模型的主干网络
        model_ref = model.module if hasattr(model, 'module') else model
        model_ref.backbone = standard_backbone
        
        # 别忘了 EMA 模型也需要保持结构一致
        ema_m = ModelEma(model, args.ema_decay)
    # =====================================================================
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))
            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dict Found!!!")
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset, val_dataset = get_datasets(args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # 注意：确保 Dataset 已经修改为返回 (image, target, index)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        # [修改] 接收元组，提取 mAUC
        metrics_res, _ = validate(val_loader, model, criterion, args, logger)
        mAUC = metrics_res['mAUC']
        logger.info(' * mAUC {mAUC:.5f}'.format(mAUC=mAUC))
        # ================= [新增] 评估模式下的每一类 AUROC 打印 =================
        if 'auc_list' in metrics_res:
            auc_list = metrics_res['auc_list']
            # 尝试获取类别名称
            class_names = getattr(val_dataset, 'classes', [])
            if not class_names or len(class_names) != len(auc_list):
                class_names = [f"Class_{i}" for i in range(len(auc_list))]
            
            logger.info("-" * 40)
            logger.info("   >>> Per-Class AUROC Details (Evaluation Mode) <<<")
            logger.info(f"   {'Class Name':<25} | {'AUROC':<10}")
            logger.info("-" * 40)
            
            for i, score in enumerate(auc_list):
                c_name = class_names[i]
                if score == -1.0:
                    score_str = "N/A"
                else:
                    score_str = f"{score:.4f}"
                logger.info(f"   {c_name:<25} | {score_str}")
            logger.info("-" * 40)
        # ======================================================================
        return

    # --- [新增] 初始化 SpliceMix ---
    #冗余代码
    if args.splicemix_mode == 'SpliceMix-CL':
        splicemix_obj = SpliceMix(mode='SpliceMix', grids=['2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    else:
        splicemix_obj = SpliceMix(mode='SpliceMix', grids=['2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    splicemix_augmentor = splicemix_obj.mixer
    # if args.splicemix_mode == 'SpliceMix-CL':
    #     splicemix_obj = SpliceMix(mode='SpliceMix', grids=['1x2','2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    # else:
    #     splicemix_obj = SpliceMix(mode='SpliceMix', grids=['1x2','2x2'], n_grids=[0], mix_prob=args.splicemix_prob)
    # splicemix_augmentor = splicemix_obj.mixer
    # --- [新增] 初始化 RoLT Handler ---
    # args.num_class 和 args.hidden_dim 必须正确设置
    rolt_handler = RoLT_Handler(args, model, train_loader, args.num_class, args.hidden_dim)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAUCs = AverageMeter('mAUC', ':5.5f', val_only=True)
    
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAUCs],
        prefix='=> Test Epoch: ')

    # -------------------------------------------------------------------------
    # [修改] 动态选择学习率调度器 (Scheduler)
    # -------------------------------------------------------------------------
    # args.step_per_batch 是一个我们动态添加的属性 (Flag)，
    # 用于告诉后面的 train 函数：这个调度器是该每个 Batch 更新 (如 OneCycle)，还是每个 Epoch 更新 (如 StepLR)。
    # 1. 动态获取优化器中每个组已经设定好的 lr (包含 backbone 的 0.1 倍逻辑)
    # 这样 backbone 组的 max_lr 就是 0.5e-5，其他组是 5e-5
    found_lrs = [group['lr'] for group in optimizer.param_groups]
    if args.scheduler == 'OneCycle':
        # [原有逻辑] OneCycleLR: 需要在每个 Batch 结束后更新 (step)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=found_lrs, # <--- 修正：传入列表，尊重差异
            steps_per_epoch=len(train_loader), 
            epochs=args.epochs, 
            pct_start=0.5
        )
        args.step_per_batch = True # 标记：Batch 级更新

    elif args.scheduler == 'StepLR':
        # [新增逻辑] StepLR: 只需要在每个 Epoch 结束后更新 (step)
        # step_size: 多少个 epoch 衰减一次
        # gamma: 衰减倍率 (例如 0.1 表示变为原来的 10%)
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.gamma
        )
        args.step_per_batch = False # 标记：Epoch 级更新

    else:
        # 防御性编程：如果传入了未实现的调度器名称，抛出错误
        raise NotImplementedError("Scheduler {} not implemented".format(args.scheduler))
    
    # 记录一些统计变量
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAUCs = AverageMeter('mAUC', ':5.5f', val_only=True)
    best_epoch = -1
    end = time.time()
    best_epoch = -1
    # [新增] 在循环外初始化变量，防止作用域报错
    clean_mask_dict = {}
    soft_label_dict = {}
    # ================= [替换] 整个训练循环 =================
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        # [核心修复] 逻辑：如果是 Stage 1，或者（是 Stage 2 但字典为空，说明是刚 Resume），必须运行 step
        need_run_rolt = (epoch <= args.splicemix_start_epoch) or (not clean_mask_dict)
        if need_run_rolt:
            clean_mask_dict, soft_label_dict = rolt_handler.step(epoch)
        # --- [新增修改] 到达指定 epoch，保存并自杀退出 ---
        if epoch == args.splicemix_start_epoch:
            if args.rank == 0:
                logger.info(f" >>> [Stage 1 Complete] Epoch {epoch} reached. Saving clean indices... <<<")
                
                # 遍历字典，如果值为 True (干净)，就把它的 key (索引) 存下来
                clean_indices = [idx for idx, is_clean in clean_mask_dict.items() if is_clean]
                
                save_path = os.path.join(args.output, 'clean_indices.pt')
                torch.save(clean_indices, save_path)
                logger.info(f" >>> Saved {len(clean_indices)} clean samples to {save_path} <<<")
            
            # 直接退出 for 循环，Stage 1 结束！
            break 
        # ------------------------------------------------
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        # --- [新增] Stage 2: 冻结 Q2L 相关参数 ---
        # 只有在刚好到达启动 epoch 时执行一次冻结操作即可，或者每个 epoch 检查也可以
        if args.enable_splicemix and epoch >= args.splicemix_start_epoch:
            if args.rank == 0:
                logger.info(f"[Stage 2] Freezing Q2L branch parameters (Transformer, Embed, FC)...")
            # 遍历模型参数，冻结属于 Q2L 分支的部分
            # 注意：在 DDP 下，模型包裹在 .module 中
            model_ref = model.module if hasattr(model, 'module') else model
            # 冻结列表
            modules_to_freeze = [
                model_ref.transformer, 
                model_ref.query_embed, 
                model_ref.input_proj, 
                model_ref.fc
            ]
            for module in modules_to_freeze:
                if isinstance(module, torch.nn.Parameter):
                    module.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False
            # 确保 Backbone 和 SpliceMix FC 是开启的
            # (如果你的意图是连 Backbone 也冻结，只练 SpliceMix FC，请把 backbone 也加入上面的列表)
            for param in model_ref.fc_splicemix.parameters():
                param.requires_grad = True
            # (B) [核心修改] 阶段切换逻辑：备份 -> 重置 -> 换优化器
            #     (仅在切换的那一轮 epoch == 20 执行一次)
            # =======================================================
            if epoch == args.splicemix_start_epoch:
                # 1. [Snapshot] 先保存 Stage 1 最终权重 (防止被覆盖)
                if args.rank == 0:
                    logger.info(f"Snapshot: Saving Stage 1 final checkpoint before transition...")
                    save_checkpoint({
                        'epoch': epoch, 
                        'arch': args.backbone,
                        'state_dict': model.state_dict(),
                        'best_mAUC': best_mAUC,
                        'optimizer': optimizer.state_dict(), 
                    }, is_best=False, filename=os.path.join(args.output, 'checkpoint_stage1_final.pth.tar'))
                # ---------------------------------------------------------
                # 2. [彻底分离核心] 动态替换 Backbone 为标准 ResNet50 (7x7)
                # ---------------------------------------------------------
                import torchvision
                if args.rank == 0:
                    logger.info(" >>> [Stage 2] Swapping modified Backbone with STANDARD ResNet50 (7x7 output)! <<<")
                # 获取底层模型引用
                model_ref = model.module if hasattr(model, 'module') else model
                # 加载一个原汁原味、带有 ImageNet 初始权重的标准 ResNet50
                standard_resnet = torchvision.models.resnet50(pretrained=True)
                # 剔除 standard_resnet 最后的全局平均池化层和全连接层，只保留特征提取部分
                standard_backbone = torch.nn.Sequential(*list(standard_resnet.children())[:-2])
                # 将设备转移到对应的 GPU
                device = next(model_ref.parameters()).device
                standard_backbone = standard_backbone.to(device)
                # 【偷天换日】：直接覆盖掉原有的魔改 Backbone
                model_ref.backbone = standard_backbone
                if args.rank == 0:
                    logger.info(" >>> [Stage 2] Backbone successfully replaced. Feature map is now 7x7. <<<")
                # ---------------------------------------------------------
                # 3. [Switch Optimizer] 切换优化器
                if args.rank == 0:
                    logger.info(f" >>> [Stage 2 Transition] Switching Optimizer to {args.optim_stage2} with LR: {args.lr_stage2} <<<")

                # 收集参数 (此时 Transformer 已被上面的逻辑冻结，不会被包含进来)
                # stage2_params = [p for p in model.parameters() if p.requires_grad]
                # 1. 区分参数: 将 Backbone (lr*0.1) 与 Head (lr) 分开
                backbone_params = []
                head_params = []
                for name, param in model_ref.named_parameters():
                    # 只收集需要梯度的参数 (注意: Transformer 等已被冻结，不会进入此循环)
                    if not param.requires_grad:
                        continue
                    # 依据名字分组
                    if 'backbone' in name:
                        backbone_params.append(param)
                    else:
                        head_params.append(param)

                if args.rank == 0:
                    logger.info(f"Stage 2 Optimizer Grouping: {len(backbone_params)} backbone params (lr*0.1), {len(head_params)} head params (lr).")

                # 2. 设置分组参数列表
                # 关键修复:给 Backbone 乘以 0.1 的系数
                params_group = [
                    {'params': backbone_params, 'lr': args.lr_stage2 * 0.1}, 
                    {'params': head_params, 'lr': args.lr_stage2}
                ]
                # 创建新优化器
                if args.optim_stage2 == 'AdamW':
                    optimizer = torch.optim.AdamW(params_group, lr=args.lr_stage2, weight_decay=args.weight_decay)
                elif args.optim_stage2 == 'SGD':
                    optimizer = torch.optim.SGD(params_group, lr=args.lr_stage2, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
                elif args.optim_stage2 == 'Adam_twd':
                     optimizer = torch.optim.Adam(params_group, lr=args.lr_stage2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

                stage2_sched_name = args.scheduler_stage2 if args.scheduler_stage2 else args.scheduler
                
                remaining_epochs = args.epochs - epoch

                if stage2_sched_name == 'OneCycle':
                    scheduler = lr_scheduler.OneCycleLR(
                        optimizer, max_lr=args.lr_stage2, 
                        steps_per_epoch=len(train_loader), 
                        epochs=remaining_epochs, 
                        pct_start=0.2
                    )
                    args.step_per_batch = True
                elif stage2_sched_name == 'StepLR':
                    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
                    args.step_per_batch = False
                if args.rank == 0:
                    logger.info(f" >>> [Stage 2 Transition] Optimizer reset complete. Active params: {len(params_group[0]['params']) + len(params_group[1]['params'])}. <<<")
                # =========================================================
                # 4. [Sync EMA] 重新初始化 EMA 模型，使其与新的 Backbone 结构保持一致
                # =========================================================
                ema_m = ModelEma(model, args.ema_decay)
                if args.rank == 0:
                    logger.info(" >>> [Stage 2 Transition] Re-initialized EMA model to match new backbone structure. <<<")
        n_total = len(train_dataset)
        if len(clean_mask_dict) == 0:
            n_clean = n_total
        else:
            # 统计 False (Noisy) 的数量，剩下的就是 Clean
            n_noisy_count = sum(1 for v in clean_mask_dict.values() if v is False)
            n_clean = n_total - n_noisy_count
        # --- 2. 训练阶段 ---
        train_start = time.time()
        
        # 调用 train 函数
        loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger,
                     splicemix_augmentor, clean_mask_dict, soft_label_dict)
        
        train_duration = sec_to_str(time.time() - train_start)

        # [关键] StepLR 必须在 Epoch 结束时更新
        # args.step_per_batch 是你在调度器部分定义的变量
        if not getattr(args, 'step_per_batch', True):
            scheduler.step()

        # 获取学习率 (用于打印)
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        lr_str = " ".join([f"{lr:.5g}" for lr in current_lrs])
        
        # [日志格式 1] Train Log
        # 样例: [Epoch 17, lr[0.005 0.05 ]] [Train] time:02:34s, loss: 0.1927 .
        if args.rank == 0:
            logger.info(f"[Epoch {epoch}, lr[{lr_str} ]] [Train] time:{train_duration}s, loss: {loss:.4f} .")
        
        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', current_lrs[0], epoch)

        # --- 3. 验证阶段 ---
        if epoch % args.val_interval == 0:
            val_start = time.time()
            
            # [修改4] 调用 validate 时，显式传入 epoch=epoch
            metrics_res, val_loss = validate(val_loader, model, criterion, args, logger, epoch=epoch)
            
            # EMA 模型验证 (同样传入 epoch)
            metrics_res_ema, val_loss_ema = validate(val_loader, ema_m.module, criterion, args, logger, epoch=epoch)
            
            val_duration = sec_to_str(time.time() - val_start)

            # 日志里同时打出两者的 mAUC
            if args.rank == 0:
                logger.info(f" -> Main Model mAUC: {metrics_res['mAUC']:.4f}")
                logger.info(f" -> EMA Model mAUC:  {metrics_res_ema['mAUC']:.4f}")
            # 提取指标
            mAUC = metrics_res['mAUC']
            mi_f1, ma_f1 = metrics_res['micro_F1'], metrics_res['macro_F1']
            mi_p, ma_p = metrics_res['micro_P'], metrics_res['macro_P']
            mi_r, ma_r = metrics_res['micro_R'], metrics_res['macro_R']

            losses.update(val_loss)
            mAUCs.update(mAUC)

            # ================= [这是你要修改/替换的核心部分] =================
            # 逻辑：Stage 1 仅做预热，不评选 Best；Stage 2 才开始评选 Best
            if epoch < args.splicemix_start_epoch:
                # Stage 1: 强制 is_best 为 False
                # 这样就不会生成 model_best.pth.tar，只保存 checkpoint.pth.tar
                is_best = False
            else:
                # Stage 2: 正常的最佳模型评选逻辑
                is_best = mAUC > best_mAUC
                if is_best:
                    best_epoch = epoch
                    best_mAUC = mAUC # 只有在 Stage 2 才更新全局最佳指标
            # ================= [修改结束] =================

            # [日志格式 2] Test Log
            timestamp = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
            log_prefix = f"[{args.rank}|{timestamp}]"
            
            if args.rank == 0:
                logger.info(
                    f"{log_prefix}: [Test] time: {val_duration}s, loss: {val_loss:.4f}, "
                    f"mAUC: {mAUC:.4f}, miF1: {mi_f1:.4f}, maF1: {ma_f1:.4f}, "
                    f"miP: {mi_p:.4f}, maP: {ma_p:.4f} ."
                )
            # [日志格式 3] Best Result Log (这里也要配合修改一下，让日志更清晰)
            if args.rank == 0:
                if epoch < args.splicemix_start_epoch:
                     logger.info(f"{log_prefix}: --[Stage 1 Warmup] (No Best Selection)")
                else:
                     logger.info(f"{log_prefix}: --[Test-best] (E{best_epoch}), mAUC: {best_mAUC:.4f}")

            if summary_writer:
                summary_writer.add_scalar('val_mAUC', mAUC, epoch)
                summary_writer.add_scalar('val_loss', val_loss, epoch)

            # --- 4. 写入 log.txt ---
            if args.rank == 0:
                log_txt_path = os.path.join(args.output, 'log.txt')
                
                lr_backbone = current_lrs[0]
                lr_head = current_lrs[1] if len(current_lrs) > 1 else current_lrs[0]
                
                with open(log_txt_path, 'a') as f:
                    # [新增] 每次写数据前，都先写一遍表头
                    # 建议在表头前加个换行符 \n，让视觉上更清晰
                    header = (
                        "\nEpoch\tTrain_Loss\tVal_Loss\tVal_mAUC\t"
                        "mi_F1\tma_F1\tmi_R\tma_R\tmi_P\tma_P\t"
                        "Clean_S\tTotal_S\tLR_Backbone\tLR_Head\n"
                    )
                    f.write(header)
                    
                    # 写数据
                    log_line = (
                        f"{epoch}\t{loss:.5f}\t{val_loss:.5f}\t{mAUC:.5f}\t"
                        f"{mi_f1:.5f}\t{ma_f1:.5f}\t{mi_r:.5f}\t{ma_r:.5f}\t{mi_p:.5f}\t{ma_p:.5f}\t"
                        f"{n_clean}\t{n_total}\t{lr_backbone:.8f}\t{lr_head:.8f}\n"
                    )
                    f.write(log_line)

            # 保存 Checkpoint
            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.backbone,
                    'state_dict': model.state_dict(),
                    'best_mAUC': best_mAUC,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))


def compute_consistency_target(preds_original, flag, device):
    """
    [修改说明] 性能优化版本：使用列表收集结果，避免在循环中频繁调用 torch.cat
    """
    mix_dict = flag['mix_dict']
    
    # 初始化为列表
    preds_m_r_list = []
    
    for i, (rand_ind, g_row, g_col, n_drop, drop_ind) in enumerate(zip(
        mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], 
        mix_dict['n_drops'], mix_dict['drop_inds']
    )):
        current_preds = preds_original[rand_ind]
        
        if n_drop > 0:
            if drop_ind.dim() == 1:
                mask = (drop_ind[:, None] == 1)
            else:
                mask = (drop_ind == 1)
            current_preds = torch.masked_fill(current_preds, mask, -1e3)
            
        # [修改] 存入列表，而不是直接 cat
        preds_m_r_list.append(current_preds)
        
    # [修改] 最后一次性拼接
    if len(preds_m_r_list) > 0:
        preds_m_r = torch.cat(preds_m_r_list, dim=0)
    else:
        preds_m_r = torch.tensor([], device=device)
        
    return preds_m_r

def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger, 
          splicemix_augmentor, clean_mask_dict, soft_label_dict):
    
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses = AverageMeter('Loss', ':5.3f')
    
    # 确保模型处于训练模式
    model.train()
    
    end = time.time()
    
    # 定义一致性损失函数 (BCE) - 用于 Stage 2 的 SpliceMix-CL
    consistency_criterion = torch.nn.BCEWithLogitsLoss()
    # 新增 resnet50 风格的一致性损失对象，专供 Stage 2 用
    splicemix_cl_criterion = Splicemix_CL_Loss_fn().cuda()
    # 定义阶段标志
    # Stage 1: epoch < start_epoch (Q2L Warmup)
    # Stage 2: epoch >= start_epoch (SpliceMix Only)
    is_stage2 = args.enable_splicemix and (epoch >= args.splicemix_start_epoch)

    for i, (images, target, indices) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)

        # --- A. 区分样本 ---
        # 根据 RoLT 返回的字典判断哪些样本是 Clean 的
        if len(clean_mask_dict) == 0:
            # 默认全 Clean (初始状态或字典为空时)
            batch_clean_mask = torch.ones(images.size(0), dtype=torch.bool).cuda()
        else:
            batch_clean_mask = torch.tensor([clean_mask_dict.get(idx.item(), True) for idx in indices]).bool().cuda()
        
        batch_noisy_mask = ~batch_clean_mask
        clean_idxs = torch.where(batch_clean_mask)[0]
        noisy_idxs = torch.where(batch_noisy_mask)[0]

        final_loss = torch.tensor(0.0).cuda()

        # ==========================================================
        #  Stage 2: 仅训练 SpliceMix 分支 (Backbone + GAP Head)
        # ==========================================================
        if is_stage2:
            if len(clean_idxs) > 4:
                images_clean = images[clean_idxs]
                targets_clean = target[clean_idxs]
                mixed_images_all, mixed_targets_all, flag = splicemix_augmentor(images_clean, targets_clean)
                
                # mix_ind = flag['mix_ind'] # 0 是原图, 1 是混合图
                
                with torch.cuda.amp.autocast(enabled=args.amp):
                    # 1. 前向传播：模型现在用的是您刚才换上的标准 7x7 Backbone
                    # src_all 已经是完美的 7x7 形状了
                    _, _, out_gap_all, src_all = model(mixed_images_all)
                    # ----------------------------------------------------
                    # 【核心解决：原因一】：抛弃 sqr_1 内部生成的 out_gap_all
                    # 我们手动对 7x7 的 src_all 进行“全局最大池化 (MaxPool)”
                    # ----------------------------------------------------
                    glb_max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
                    src_all_gp = glb_max_pool(src_all).flatten(1)
                    
                    # 用最大池化的特征过一遍分类头
                    fc_head = model.module.fc_splicemix if hasattr(model, 'module') else model.fc_splicemix
                    out_maxpool_all = fc_head(src_all_gp)
                    # 2. 将之前所有的 out_gap_all 替换为我们算出来的 out_maxpool_all
                    if args.splicemix_mode == 'SpliceMix-CL':
                        model_outputs = get_splicemix_outputs(model, src_all, out_maxpool_all, flag)
                    else:
                        model_outputs = out_maxpool_all
                    
                    # 3. 这里的 final_loss 终于和 resnet50 仓库毫无二致了！
                    final_loss = splicemix_cl_criterion(model_outputs, mixed_targets_all)

            # 如果本 batch 干净样本不足，final_loss 保持 0，不进行梯度更新
            else:
                final_loss = torch.tensor(0.0).cuda()

        # ==========================================================
        #  Stage 1: Q2L 原生训练 (Q2L Warmup)
        # ==========================================================
        else:
            loss_q2l_branch = 0.0
            with torch.cuda.amp.autocast(enabled=args.amp):
                # 前向传播所有原始图片
                # [关键] Stage 1 训练 Q2L Transformer，取第1个返回值 out_trans_all
                out_trans_all, _,out_splicemix, _ = model(images)
                # 2. [修改] 判断是否处于 RoLT 预热期
                #    如果当前 epoch 小于设定的启动轮数，强制进行普通的监督训练
                if epoch < args.rolt_start_epoch:
                    # --- 预热模式 (Warmup Mode) ---
                    # 忽略 RoLT 的 Mask，直接计算所有样本的 Loss
                    # 这能让模型先学到基础特征，避免一开始被错误的 RoLT 筛选带偏
                    loss_q2l_branch = criterion(out_trans_all, target)
                else :
                    valid_parts = 0
                    # C.1 干净样本 -> 使用原始硬标签 (Hard Label)
                    if len(clean_idxs) > 0:
                        loss_clean = criterion(out_trans_all[clean_idxs], target[clean_idxs])
                        loss_q2l_branch += loss_clean
                        valid_parts += 1
                    # C.2 噪声样本 -> 使用 RoLT 生成的软标签 (Soft Label)
                    if len(noisy_idxs) > 0:
                        soft_targets_list = []
                        for k in noisy_idxs:
                            global_idx = indices[k].item()
                            # 尝试取软标签，取不到则兜底使用原始标签
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

        # --- Backprop (反向传播) ---
        # 只有当 Loss > 0 时才反向传播
        if isinstance(final_loss, torch.Tensor) and final_loss.item() > 0:
            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
            
            # [重要] 梯度裁剪
            # 必须先 unscale 才能 clip
            scaler.unscale_(optimizer)
            # [修改] 差异化梯度裁剪
            if not is_stage2:
                # Stage 1 (Transformer): 必须裁剪，防爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                # pass
            else:
                # Stage 2 (CNN): 移除裁剪，让梯度自由流动
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                pass
            
            scaler.step(optimizer)
            scaler.update()
            
            # 如果使用 OneCycleLR (batch级调度)，则在此 step
            if getattr(args, 'step_per_batch', False):
                scheduler.step()

        # EMA 更新
        if epoch >= args.ema_epoch:
            ema_m.update(model)

        losses.update(final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg
@torch.no_grad()
def validate(val_loader, model, criterion, args, logger, epoch=None):  # <--- [修改1] 增加 epoch 参数默认值
    # AveragePrecisionMeter自定义类，用于累积预测结果并计算mAP/mAUC等指标
    meter = AveragePrecisionMeter(difficult_examples=False)
    losses = AverageMeter('Loss', ':5.3f') 
    
    model.eval()
    
    for i, (images, target, _) in enumerate(val_loader): 
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # 混合精度上下文
        with torch.cuda.amp.autocast(enabled=args.amp):
            # 1. 必须接收第 4 个参数 src_all (底层 7x7 特征图)
            out_logits, _, out_splicemix, src_all = model(images)
            
            if epoch is not None and epoch < args.splicemix_start_epoch:
                output = out_logits       # Stage 1: 评估 Transformer 分支
            else:
                # Stage 2: 严格对齐训练时的 MaxPool 处理！
                glb_max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))
                src_all_gp = glb_max_pool(src_all).flatten(1)
                
                # 获取对应的 SpliceMix 分类头
                fc_head = model.module.fc_splicemix if hasattr(model, 'module') else model.fc_splicemix
                output = fc_head(src_all_gp)  # 使用 MaxPool 特征进行预测
            
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            
            meter.add(output, target, filename=[]) 

    # 获取所有指标
    metrics_res = meter.compute_all_metrics()
    
    return metrics_res, losses.avg

# --- 工具类保持不变 ---
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

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
    # [修复] 无论是否是 Best，都要保存当前 checkpoint (覆盖写)，用于断点续训
    torch.save(state, filename)
    # 如果是 Best，额外备份一份
    if is_best:
        best_path = os.path.join(os.path.split(filename)[0], 'model_best.pth.tar')
        torch.save(state, best_path)

class AverageMeter(object):
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class AverageMeterHMS(AverageMeter):
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name, 
                             val=str(datetime.timedelta(seconds=int(self.val))), 
                             sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def compute_cl_loss(model, mixed_spatial_features, preds_original, flag, criterion):
    """
    仿照 SpliceMix_CL.py 实现特征切分和一致性 Loss
    :param model: Q2L 模型 (需要调用其 fc_splicemix 和 GAP)
    :param mixed_spatial_features: [B_mix, C, H, W] 混合图的空间特征
    :param preds_original: [B_orig, Num_Class] 原图的预测结果 (作为 Target)
    :param flag: SpliceMix 返回的混合信息字典
    """
    mix_dict = flag['mix_dict']
    mix_ind = flag['mix_ind'] # 这里的 mix_ind 可能需要根据你的 dataset 实现调整，参考下文
    
    # 这里的 mixed_spatial_features 对应参考代码中的 feas_m
    # 这里的 preds_original 对应参考代码中的 preds_r
    
    feas_m = mixed_spatial_features
    B, C, H, W = feas_m.shape
    
    preds_m_list = []   # 切分后的混合图预测 (Student)
    preds_r_list = []   # 对应的原图预测 (Teacher)
    
    # 遍历每一种混合模式 (1x2, 2x2 等)
    start_idx = 0
    # 注意：SpliceMix.py 的实现里，mix_dict 的元素是按顺序 append 的
    # 所以我们需要按顺序处理 mixed_spatial_features
    
    for i in range(len(mix_dict['rand_inds'])):
        rand_ind = mix_dict['rand_inds'][i] # 这一组混合涉及的原图索引
        g_row = mix_dict['rows'][i]
        g_col = mix_dict['cols'][i]
        n_drop = mix_dict['n_drops'][i]
        drop_ind = mix_dict['drop_inds'][i]
        
        # 计算这一组产生了多少张混合图
        # rand_ind 的长度是总原图数，除以网格大小 = 混合图数
        ng = len(rand_ind) // (g_row * g_col)
        
        # 取出这一批混合图的特征
        current_feas_m = feas_m[start_idx : start_idx + ng]
        start_idx += ng
        
        # --- 核心切分逻辑 (参考 SpliceMix_CL.py) ---
        # 如果特征图尺寸不能被网格整除，需要插值 (通常 ResNet50 的 7x7 或 14x14 很难被 2 整除，这是一个潜在坑)
        if H % g_row != 0 or W % g_col != 0:
             current_feas_m = torch.nn.functional.interpolate(
                 current_feas_m, 
                 (H // g_row * g_row, W // g_col * g_col), 
                 mode='bilinear', align_corners=True
             )
        
        # 将特征图切块
        # split 维度 2 (高度)
        row_chunks = current_feas_m.split(current_feas_m.shape[-2] // g_row, dim=-2) 
        chunks = []
        for rc in row_chunks:
            # split 维度 3 (宽度)
            chunks.extend(rc.split(current_feas_m.shape[-1] // g_col, dim=-1))
            
        # chunks 现在是一个列表，包含 (g_row * g_col) 个 tensor
        # 每个 tensor shape: [ng, C, h', w']
        # 我们需要把它们堆叠起来变成 [ng * grids, C, h', w']
        # 注意顺序：SpliceMix 可能是按行优先或列优先，需要核对 SpliceMix.py 的 make_grid
        # torchvision.make_grid 默认是行优先，所以这里直接 stack 再 view 应该是对的
        
        # [ng, grid_size, C, h', w'] -> [ng * grid_size, C, h', w']
        fea_m_patches = torch.stack(chunks, dim=1).view(-1, C, chunks[0].shape[-2], chunks[0].shape[-1])
        
        # --- 对切好的块做预测 ---
        # 1. GAP
        fea_m_gp = fea_m_patches.amax(dim=[2, 3]) 
        # 2. FC (使用模型的 fc_splicemix)
        pred_m = model.module.fc_splicemix(fea_m_gp) # 注意 DDP 下用 model.module
        
        # --- 处理对应的原图 Target ---
        # preds_original[rand_ind] 取出对应的原图预测
        pred_r = preds_original[rand_ind]
        
        # 处理 Drop (如果有 patch 被丢弃，mask 掉它的 Loss)
        if n_drop > 0:
             # drop_ind: [Total_Source_Images] -> 0 or 1
             mask = (drop_ind == 1)
             # 将被 drop 的位置设为 -1e3 (sigmoid 后接近 0)，或者直接在 Loss 处 mask
             # 参考代码的做法是 mask 填值
             pred_m = torch.masked_fill(pred_m, mask[:, None], -1e3)
             pred_r = torch.masked_fill(pred_r, mask[:, None], -1e3)

        preds_m_list.append(pred_m)
        preds_r_list.append(pred_r)
        
    # 拼合所有批次
    preds_m_all = torch.cat(preds_m_list, dim=0)
    preds_r_all = torch.cat(preds_r_list, dim=0)
    
    # 计算一致性 Loss
    loss_cl = criterion(preds_m_all, torch.sigmoid(preds_r_all).detach())
    
    return loss_cl
if __name__ == '__main__':
    main()