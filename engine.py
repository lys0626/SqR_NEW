import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, shutil
from torch.cuda.amp import GradScaler, autocast
from utilities_s2 import utils, metric, utils_ddp, warmup, logger
from SpliceMix import SpliceMix
import models_s2 as models
import copy, math
from models_s2.SpliceMix_CL import Loss_fn
import numpy as np
from torch.utils.data import Dataset
def _resolve_stage1_tensor_path(explicit_path, soft_label_path, filename):
    if explicit_path:
        return explicit_path
    if soft_label_path:
        return os.path.join(os.path.dirname(soft_label_path), filename)
    return ''

def _load_stage1_tensor(path, shape, default_value, dtype):
    if path and os.path.exists(path):
        tensor = torch.load(path, map_location='cpu')
        assert tensor.shape == shape, f"Stage1 tensor shape {tensor.shape} mismatch with expected {shape}: {path}"
        return tensor.to(dtype=dtype)
    return torch.full(shape, default_value, dtype=dtype)

class ReliabilitySoftLabelDatasetWrapper(Dataset):
    """
    字典兼容的数据集包装器：
    将 Stage 1 的软标签注入 data['target'] 供网络训练，
    同时将原硬标签保留为 data['target_hard'] 供指标监控。
    """
    def __init__(self, original_dataset, soft_targets_tensor, label_weight=None, label_reliability=None, fp_mask=None, fn_mask=None):
        self.dataset = original_dataset
        self.soft_targets = soft_targets_tensor.cpu() # 放在 CPU 防止主进程显存爆炸
        self.label_weight = label_weight.cpu() if label_weight is not None else torch.ones_like(self.soft_targets)
        self.label_reliability = label_reliability.cpu() if label_reliability is not None else torch.ones_like(self.soft_targets)
        self.fp_mask = fp_mask.cpu().bool() if fp_mask is not None else torch.zeros_like(self.soft_targets, dtype=torch.bool)
        self.fn_mask = fn_mask.cpu().bool() if fn_mask is not None else torch.zeros_like(self.soft_targets, dtype=torch.bool)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 1. 获取原始数据字典
        data = self.dataset[index]
        
        # 2. 保留真实的硬标签用于指标统计（深度拷贝防篡改）
        if 'target_hard' not in data:
            data['target_hard'] = data['target'].clone() if torch.is_tensor(data['target']) else data['target']
        
        # 3. 将软标签注入 target 键，供网络反向传播
        data['target'] = self.soft_targets[index].clone()
        data['label_reliability'] = self.label_reliability[index].clone()
        data['fp_mask'] = self.fp_mask[index].clone()
        data['fn_mask'] = self.fn_mask[index].clone()
        data['label_weight'] = self.label_weight[index].clone()
        data['sample_index'] = index
        
        return data
class ModelEMA:
    """ 
    模型参数的指数移动平均 (Exponential Moving Average)
    有效平滑训练轨迹，提升对噪声标签的鲁棒性
    """
    def __init__(self, model, decay=0.999, updates=0):
        # 提取底层的 model，剥离 DDP 的 module 壳
        self.ema = copy.deepcopy(model.module if hasattr(model, 'module') else model).eval()
        self.updates = updates
        # 带有 warmup 的衰减函数，前期更新快，后期稳定
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()
class Engine(object):
    def __init__(self, args):
        super(Engine, self).__init__()
        self.args = args
        self.result = {}
        self.result['train'] = {'epoch': [], 'lr': [], 'loss': []}
        self.result['val'] = {'epoch': [], 'loss': [], 'mAUC': [], 'micro_F1': []}
        # 这里的 val_best 记录 mAUC
        self.result['val_best'] = {'epoch': 0, 'loss': -1., 'mAUC': -1., 'metrics': {}}
        # --- 新增：EMA 模型的追踪 ---
        self.result['ema_val'] = {'epoch': [], 'loss': [], 'mAUC': [], 'micro_F1': []}
        self.result['ema_val_best'] = {'epoch': 0, 'loss': -1., 'mAUC': -1., 'metrics': {}}
        # 测试集 (Test) 记录器
        self.result['test'] = {'epoch': [], 'loss': [], 'mAUC': [], 'micro_F1': []}
        self.result['test_best'] = {'epoch': 0, 'loss': -1., 'mAUC': -1., 'metrics': {}}
        self.result['ema_test'] = {'epoch': [], 'loss': [], 'mAUC': [], 'micro_F1': []}
        self.result['ema_test_best'] = {'epoch': 0, 'loss': -1., 'mAUC': -1., 'metrics': {}}
        self.meter = {}
        self.reset_meters()

        self.rank = utils_ddp.get_rank()
        
        method_name = "baseline"
        if self.args.model == 'SpliceMix_CL':
            method_name = 'splicemix-CL'
        elif 'SpliceMix' in self.args.mixer:
            method_name = 'splicemix'

        log_file = f'{self.args.data_set}_{method_name}_{self.args.start_time}.log'
        
        self.logger = logger.setup_logger(os.path.join(self.args.save_path, 'log', log_file), self.rank)
        self.logger.info(args)
        self.init()

    def init(self):
        train_set, val_set, test_set, self.args.num_classes = utils.get_dataset(self.args)

        # ==========================================
        # 🌟 核心接入：覆盖训练集的 Target 为软标签
        # ==========================================
        # 假设在 stage2_main.py 中传入了 --soft_label_path
        soft_label_path = getattr(self.args, 'soft_label_path', '')
        
        #不传入软标签路径，模型会默认采用硬标签
        if soft_label_path and os.path.exists(soft_label_path):
            self.logger.info(f" >>> Loading Asymmetric Soft Targets from: {soft_label_path}")
            
            soft_targets_tensor = torch.load(soft_label_path, map_location='cpu')
            
            # 安全校验：确保软标签数量与训练集长度一致
            assert soft_targets_tensor.shape[0] == len(train_set), \
                f"🚨 Error: Soft targets length ({soft_targets_tensor.shape[0]}) mismatch with train_set ({len(train_set)})!"
            label_weight = None
            label_reliability = None
            fp_mask = None
            fn_mask = None
            if getattr(self.args, 'use_stage1_reliability', False):
                shape = soft_targets_tensor.shape
                reliability_path = _resolve_stage1_tensor_path(getattr(self.args, 'label_reliability_path', ''), soft_label_path, 'label_reliability.pt')
                fp_path = _resolve_stage1_tensor_path(getattr(self.args, 'fp_mask_path', ''), soft_label_path, 'fp_mask.pt')
                fn_path = _resolve_stage1_tensor_path(getattr(self.args, 'fn_mask_path', ''), soft_label_path, 'fn_mask.pt')
                hard_clean_path = _resolve_stage1_tensor_path(getattr(self.args, 'hard_clean_mask_path', ''), soft_label_path, 'hard_clean_mask.pt')

                label_reliability = _load_stage1_tensor(reliability_path, shape, 1.0, torch.float32).clamp(0.0, 1.0)
                fp_mask = _load_stage1_tensor(fp_path, shape, False, torch.bool)
                fn_mask = _load_stage1_tensor(fn_path, shape, False, torch.bool)
                hard_clean_mask = _load_stage1_tensor(hard_clean_path, shape, False, torch.bool)

                label_weight = label_reliability.clamp(
                    min=getattr(self.args, 'reliability_min_weight', 0.2),
                    max=1.0,
                )
                fp_cap = torch.full_like(label_weight, getattr(self.args, 'fp_loss_weight', 0.3))
                fn_floor = torch.full_like(label_weight, getattr(self.args, 'fn_loss_weight', 0.7))
                label_weight = torch.where(fp_mask, torch.minimum(label_weight, fp_cap), label_weight)
                label_weight = torch.where(fn_mask, torch.maximum(label_weight, fn_floor), label_weight)
                label_weight = torch.where(hard_clean_mask, torch.ones_like(label_weight), label_weight)
                self.logger.info(f" >>> Loaded Stage 1 reliability weights: {reliability_path}")
            train_set = ReliabilitySoftLabelDatasetWrapper(
                train_set,
                soft_targets_tensor,
                label_weight=label_weight,
                label_reliability=label_reliability,
                fp_mask=fp_mask,
                fn_mask=fn_mask,
            )
            self.logger.info(" >>> 🎯 Successfully wrapped train_set with Stage 1 Soft Labels!")
        else:
            self.logger.warning(" !!! No soft labels found. Training with ORIGINAL HARD LABELS !!!")
        # ==========================================

        self.dataset = {'train': train_set, 'val': val_set, 'test': test_set}
        self.scaler = GradScaler(enabled=not self.args.disable_amp)

        args = {}
        #getattr动态获取模块
        self.model = getattr(models, self.args.model).model(self.args.num_classes, args=args).to(self.rank)
        self.ema = ModelEMA(self.model)
        self.optimizer = utils.get_optimizer(self.args, self.model)
        self.loss_fn = getattr(models, self.args.model).Loss_fn().to(self.rank)
        
        self.train_loader, self.val_loader, self.test_loader = utils.get_dataloader(
            train_set=train_set, val_set=val_set, test_set=test_set, args=self.args)
        if self.args.warmup_epochs > 0:
            self.warmup_scheduler = warmup.WarmUpLR(self.optimizer,
                                                    total_iters=len(self.train_loader) * self.args.warmup_epochs)
        self.lr_scheduler = utils.get_lr_scheduler(self.args, self.optimizer)
        self.load_checkpoint()

        if self.args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        
        if 'SpliceMix' in self.args.mixer:
            self.mixer = SpliceMix(mode=self.args.mixer, grids=self.args.grids,
                                             n_grids=self.args.n_grids, mix_prob=self.args.Sprob).mixer

    def train(self):
        if self.args.start_epoch == 0:
            self.args.start_epoch = 1
        for epoch in range(self.args.start_epoch, self.args.epochs+1):
            train_loader = self.train_loader
            self.model.train()
            self.on_start_epoch(epoch)
            if self.args.distributed:
                train_loader.sampler.set_epoch(epoch)
            torch.cuda.empty_cache()

            for i, data in enumerate(train_loader):
                inputs, targets, targets_gt, file_name, label_weight = self.on_start_batch(data)
                outputs, loss = self.on_forward(inputs, targets, file_name, is_train=True, label_weight=label_weight)
                self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

            self.on_end_epoch(is_train=True, result=self.result['train'])
            self.lr_scheduler.step()

            if self.args.evaluate > 0 and ((epoch % self.args.evaluate == 0) or epoch == 1):
                self.evaluate(epoch=epoch)

    def evaluate(self, epoch=0):
        torch.cuda.empty_cache()
        
        # 定义需要评估的模型列表
        eval_pairs = []
        # 如果是训练阶段，才需要评估普通模型
        if self.args.is_train:
            # 训练阶段：普通模型和 EMA 模型都要评估以对比记录
            eval_pairs.append({'model': self.model, 'loader': self.val_loader, 'res': self.result['val'], 'best': self.result['val_best'], 'is_ema': False, 'tag': 'Val'})
            eval_pairs.append({'model': self.model, 'loader': self.test_loader, 'res': self.result['test'], 'best': self.result['test_best'], 'is_ema': False, 'tag': 'Test'})
            if hasattr(self, 'ema'):
                eval_pairs.append({'model': self.ema.ema, 'loader': self.val_loader, 'res': self.result['ema_val'], 'best': self.result['ema_val_best'], 'is_ema': True, 'tag': 'EMA-Val'})
                eval_pairs.append({'model': self.ema.ema, 'loader': self.test_loader, 'res': self.result['ema_test'], 'best': self.result['ema_test_best'], 'is_ema': True, 'tag': 'EMA-Test'})
        else:
            # 纯测试阶段：根据加载的权重类型，只评估对应的一个模型
            is_ema_loaded = getattr(self, 'loaded_is_ema', False)
            
            # 双重保险：如果 checkpoint 字典里写了是 EMA，或者文件名里带了 'EMA'
            # if is_ema_loaded or 'EMA' in self.args.resume:
            if is_ema_loaded or 'EMA' in os.path.basename(self.args.resume):
                eval_pairs.append({'model': self.ema.ema, 'loader': self.test_loader, 'res': self.result['ema_test'], 'best': self.result['ema_test_best'], 'is_ema': True, 'tag': 'EMA-Test'})
            else:
                eval_pairs.append({'model': self.model, 'loader': self.test_loader, 'res': self.result['test'], 'best': self.result['test_best'], 'is_ema': False, 'tag': 'Test'})
        for item in eval_pairs:
            m = item['model']
            m.eval()
            self.on_start_epoch(epoch) 
            
            self.logger.info(f"==> Evaluating {item['tag']} Set...")
            
            for i, data in enumerate(item['loader']):
                inputs, targets, targets_gt, file_name, label_weight = self.on_start_batch(data)
                with torch.no_grad():
                    with autocast(enabled=not self.args.disable_amp):
                        if not item['is_ema']:
                            outputs, loss = self.on_forward(inputs, targets, file_name, is_train=False, label_weight=label_weight)
                        else:
                            outputs = m(inputs)
                            if isinstance(outputs, tuple): outputs = outputs[0]
                            if isinstance(outputs, dict): outputs = outputs['logits_mixed']
                            loss = self.loss_fn(outputs, targets)
                
                outputs = outputs[:inputs.shape[0]].data
                self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

            # 传入 eval_tag，用于精准控制日志和权重保存
            self.on_end_epoch(is_train=False, result=item['res'], result_best=item['best'], is_ema=item['is_ema'], eval_tag=item['tag'])
    def reliability_weighted_bce(self, logits, targets, label_weight):
        loss_mat = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        if label_weight is None:
            label_weight = torch.ones_like(targets)
        label_weight = label_weight.to(loss_mat.device).float()
        return (loss_mat * label_weight).sum() / label_weight.sum().clamp_min(1.0)

    def reliability_weighted_loss(self, outputs, targets, label_weight):
        if isinstance(outputs, tuple) and len(outputs) == 5:
            preds_all, preds_mix, preds_m_r, targets_all, weights_all = outputs
            loss_bce = self.reliability_weighted_bce(preds_all, targets_all, weights_all)
            loss_cl = F.binary_cross_entropy_with_logits(preds_mix, preds_m_r.sigmoid().detach())
            return loss_bce + loss_cl
        if isinstance(outputs, tuple) and len(outputs) == 3:
            preds_mix, targets_mix, weights_mix = outputs
            return self.reliability_weighted_bce(preds_mix, targets_mix, weights_mix)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            preds_mix, targets_mix = outputs
            return self.reliability_weighted_bce(preds_mix, targets_mix, torch.ones_like(targets_mix))
        return self.reliability_weighted_bce(outputs, targets, label_weight)

    def on_forward(self, inputs, targets, file_name, is_train, label_weight=None):
        # ==============================================================
        # 1. 动态计算训练进度比例 progress_ratio (0.0 到 1.0 之间)
        # 如果是纯测试调用没有 epoch 属性，默认取 1.0 (最严苛把关模式)
        # ==============================================================
        current_epoch = getattr(self, 'epoch', self.args.epochs)
        progress_ratio = current_epoch / self.args.epochs if self.args.epochs > 0 else 1.0
        args = {}
        if is_train:
            with autocast(enabled=not self.args.disable_amp):
                # ---------------- 核心修改：移除外部图片拼接 ----------------
                # --- 修改这里：只要使用了 SpliceMix，就把参数传进去 ---
                # 这样无论是 ResNet_50 还是 SpliceMix_CL，都能接收到 mixer
                if 'SpliceMix' in self.args.mixer: 
                    args = {
                        'mixer': self.mixer,
                        'targets': targets,
                        'label_weight': label_weight
                    }
                
                outputs = self.model(inputs, args)
                #原有的BCE LOSS
                loss = self.reliability_weighted_loss(outputs, targets, label_weight)

                # #下面是采用动态 ASL loss
                # # 2. 将 progress_ratio 注入到损失函数中！
                # # loss = self.loss_fn(outputs, targets, progress_ratio=progress_ratio)
                # # ==========================================
                # # 【修改点 1】：自适应兼容 Loss 传参 (训练阶段)
                # # ==========================================
                # try:
                #     # 尝试调用支持 D-ASL 的新版 Loss (带进度参数)
                #     loss = self.loss_fn(outputs, targets, progress_ratio=progress_ratio)
                # except TypeError as e:
                #     # 如果报错提示不支持 progress_ratio，则自动回退到普通 Loss 计算
                #     if 'progress_ratio' in str(e):
                #         loss = self.loss_fn(outputs, targets)
                #     else:
                #         raise e # 抛出其他非参数类型的真实错误
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.args.disable_amp:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # --- [关键] 更新 EMA 权重 ---
            if hasattr(self, 'ema'):
                self.ema.update(self.model)
            if self.args.warmup_epochs > 0 and self.epoch <= self.args.warmup_epochs:
                self.warmup_scheduler.step()
        else:
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_amp):
                    outputs = self.model(inputs, args)
                    loss = self.loss_fn(outputs, targets)

                    # # 测试验证时，同样注入进度 (此时等于 1.0)
                    # # loss = self.loss_fn(outputs, targets, progress_ratio=progress_ratio)


                    # # ==========================================
                    # # 【修改点 2】：自适应兼容 Loss 传参 (测试验证阶段)
                    # # ==========================================
                    # try:
                    #     loss = self.loss_fn(outputs, targets, progress_ratio=progress_ratio)
                    # except TypeError as e:
                    #     if 'progress_ratio' in str(e):
                    #         loss = self.loss_fn(outputs, targets)
                    #     else:
                    #         raise e
        outputs = outputs[0][:inputs.shape[0]].data if type(outputs) == tuple else outputs[:inputs.shape[0]].data
        return outputs, loss

    def on_start_batch(self, data):
        inputs = data['image'].to(self.rank)
        file_name = data['name']
        
        # 1. targets 用于喂给 Loss 计算（被 Wrapper 替换成了软标签）
        targets = data['target'].clone().to(self.rank)
        targets[targets == -1] = 0 # 兜底机制，消除未标注数据
        
        # 2. targets_gt (Ground Truth) 用于喂给 Meter 计算指标
        # 如果包装器保存了硬标签，就用硬标签；如果是测试集没被包装，就回退用 target


        targets_gt = data.get('target_hard', data['target']).clone().to(self.rank)
        targets_gt[targets_gt == -1] = 0

        label_weight = data.get('label_weight', torch.ones_like(data['target'])).clone().to(self.rank)
        label_weight = label_weight.float()

        return inputs, targets, targets_gt, file_name, label_weight

    def on_end_batch(self, outputs, targets_gt, loss, image_name=''):
        bs = self.args.batch_size
        if self.args.distributed:
            outputs = utils_ddp.distributed_concat(outputs.detach(), bs)
            targets_gt = utils_ddp.distributed_concat(targets_gt.detach().to(self.rank), bs)
            loss_all = utils_ddp.distributed_concat(loss.detach().unsqueeze(0), utils_ddp.get_world_size())
        else:
            loss_all = loss.detach().cpu().mean()

        self.meter['loss'].add(loss.cpu())
        if utils_ddp.is_main_process():
            self.meter['loss_all'].add(loss_all.detach().cpu().mean())
            self.meter['ap'].add(outputs.detach().cpu(), targets_gt.cpu(), image_name)

    def on_start_epoch(self, epoch):
        self.epoch = epoch
        self.epoch_time = time.time()
        self.reset_meters()
    def on_end_epoch(self, is_train, result, result_best=None, is_ema=False,eval_tag='Val'):
        self.lr_curr = utils.get_learning_rate(self.optimizer)
        self.epoch_time = time.time() - self.epoch_time
        meter = self.meter
        loss = meter['loss'].average()
        
        metrics_res = {}

        if utils_ddp.is_main_process():
            loss_all = meter['loss_all'].average()
            
            # --- 只有验证时计算全部医学指标 (mAUC, F1, etc.) ---
            if not is_train:
                metrics_res = meter['ap'].compute_all_metrics()
            else:
                metrics_res = {} 
        else:
            loss_all = torch.tensor(-1)
            metrics_res = {}

        if self.args.distributed:
            utils_ddp.barrier()

        # 记录基础数据
        result['epoch'].append(self.epoch)
        result['loss'].append(loss_all.item())
        if 'mAUC' in metrics_res:
            result['mAUC'].append(metrics_res['mAUC'])
        if 'micro_F1' in metrics_res:
            result['micro_F1'].append(metrics_res['micro_F1'])

        is_best = False
        
        # --- 格式化日志字符串 ---
        str_metrics = ""
        # 定义日志前缀
        # log_tag = "Train" if is_train else ("EMA-Test" if is_ema else "Test")
        log_tag = "Train" if is_train else eval_tag
        if not is_train and 'mAUC' in metrics_res:
            str_metrics = f"mAUC: {metrics_res['mAUC']:.4f}, "
            str_metrics += (
                f"miF1: {metrics_res.get('micro_F1', 0):.4f}, maF1: {metrics_res.get('macro_F1', 0):.4f}, "
                f"miR:  {metrics_res.get('micro_R', 0):.4f}, maR: {metrics_res.get('macro_R', 0):.4f}, " 
                f"miP:  {metrics_res.get('micro_P', 0):.4f}, maP: {metrics_res.get('macro_P', 0):.4f}"
            )
            
            # --- 打印每个具体疾病类别的 AUROC (Per-Class) ---
            if 'auc_list' in metrics_res:
                auc_list = metrics_res['auc_list']
                # 尝试从 dataset 中获取疾病名称，增加 EMA 模式下的容错
                target_ds = self.dataset['test']
                class_names = getattr(target_ds, 'eval_classes', None)
                if not class_names or len(class_names) != len(auc_list):
                    class_names = getattr(target_ds, 'classes', [f"Class_{i}" for i in range(len(auc_list))])
                per_class_str = ", ".join([
                    f"{name}: {auc:.4f}" if auc != -1.0 else f"{name}: N/A" 
                    for name, auc in zip(class_names, auc_list)
                ])
                self.logger.info(f"[{log_tag} Per-Class AUC] {per_class_str}")

        if isinstance(self.lr_curr, (list, tuple, np.ndarray)):
            lr_str = "[" + ", ".join([f"{float(lr):.6f}" for lr in self.lr_curr]) + "]"
        else:
            lr_str = f"{float(self.lr_curr):.6f}"

        # 打印 Epoch 总结日志 (加入 float(loss) 防止 loss 也是单个元素的 array 导致报错)
        if is_train:
            str_log = f'[Epoch {self.epoch}, lr: {lr_str}] [{log_tag}] time:{utils.strftime(self.epoch_time)}s, loss: {float(loss):.4f} .'
        else:
            str_log = f'[{log_tag}] time: {utils.strftime(self.epoch_time)}s, loss: {float(loss):.4f}, {str_metrics} .'
        
        self.logger.info(str_log)

        # --- Best Model 判定 (基于 mAUC) ---
        # ========================================================
        # 🌟 [核心修改 3]: Best Model 判定 (CheXpert 强制看 mAUC_5)
        # ========================================================
        if not is_train:
            current_target_auc = metrics_res.get('mAUC', 0.0)
            best_metric_name = "mAUC"
            if result_best['mAUC'] < current_target_auc:
                is_best = True
                result_best['mAUC'] = current_target_auc
                result_best['epoch'] = self.epoch
                result_best['loss'] = loss
                result_best['metrics'] = metrics_res

            str_best = f"--[{log_tag}-best] (E{result_best['epoch']}), {best_metric_name}: {result_best['mAUC']:.4f}"
            self.logger.info(str_best)

        if not is_train and self.args.evaluate != 0 and utils_ddp.is_main_process():
            if 'Val' in eval_tag: 
                self.save_checkpoint(is_best=is_best, is_ema=is_ema)
        if self.args.distributed:
            utils_ddp.barrier()

    # --- save_checkpoint ---
    def save_checkpoint(self, is_best, is_ema=False):
        # 提取当前对应模型的权重和结果
        if is_ema:
            state_dict = self.ema.ema.state_dict()
            res_best = self.result['ema_val_best']
        else:
            state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            res_best = self.result['val_best']

        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': state_dict,
            'mAUC_best': res_best['mAUC'],
            'result': self.result,
            'args': self.args,
            'is_ema': is_ema
        }

        # 文件命名逻辑
        prefix = "EMA_" if is_ema else ""
        
        # 只有普通模型保存 ChkpotLast
        if not is_ema:
            last_path = os.path.join(self.args.save_path, f'ChkpotLast_{self.args.model}.pt')
            torch.save(checkpoint, last_path)

        # 只要是各自的最佳 mAUC，就保存对应的 Best 权重
        if is_best:
            best_fn = f'ChkpotBest_{prefix}{self.args.model}.pt'
            best_path = os.path.join(self.args.save_path, best_fn)
            torch.save(checkpoint, best_path)
            self.logger.info(f"==> Saved Best {prefix}Model to {best_path}")

    def load_checkpoint(self):
        if self.args.resume == '':
            return
        else:
            file = self.args.resume
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            # 加载文件
            checkpoint = torch.load(file, map_location=map_location)
        # --- 新增：记录当前加载的权重类型 ---
        self.loaded_is_ema = checkpoint.get('is_ema', False)
        # ---------------------------------
        try:
            if self.args.start_epoch == 0:
                self.args.start_epoch = checkpoint['epoch'] + 1
            
            # --- 加一行判断：只在训练时恢复历史指标字典 ---
            if self.args.is_train:  
                self.result = checkpoint['result']
        except:
            pass

        # --- 修正后的加载逻辑 ---
        try:
            # 1. 确定 state_dict 在哪里
            if 'model_state_dict' in checkpoint:
                loaded_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                loaded_dict = checkpoint['state_dict']
            else:
                loaded_dict = checkpoint

            # 2. 处理键名 (去除 module. 前缀以匹配单机模型)
            new_state_dict = {}
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            stage1_len = len(base_model.stage1) if hasattr(base_model, 'stage1') else 0
            
            for k, v in loaded_dict.items():
                name = k[7:] if k.startswith('module.') else k
                
                # 兼容旧版本 backbone 权重
                if name.startswith('backbone.') and stage1_len > 0:
                    parts = name.split('.')
                    idx = int(parts[1])
                    if idx < stage1_len:
                        name = f"stage1.{idx}." + ".".join(parts[2:])
                    else:
                        name = f"stage2.{idx - stage1_len}." + ".".join(parts[2:])
                        
                new_state_dict[name] = v
            
            # 3. 加载权重 (strict=False 允许稍微的不匹配，但关键是不要主动过滤 cls)
            # 这会将训练好的 cls.weight 和 cls.bias 正确加载进去
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            # ---------------- 核心修复点 ----------------
            # 将读取到的权重同步加载到 EMA 模型中，防止其保持随机初始化状态
            if hasattr(self, 'ema'):
                self.ema.ema.load_state_dict(new_state_dict, strict=False)
            # ------------------------------------------
            self.logger.info(f"==> Loaded checkpoint from {file}")
            self.logger.info(f"    Missing keys: {msg.missing_keys}")
            self.logger.info(f"    Unexpected keys: {msg.unexpected_keys}")

        except Exception as e:
            self.logger.info(f"==> Failed to load checkpoint: {e}")

    def reset_meters(self):
        self.meter['loss'] = metric.AverageMeter('loss')
        self.meter['loss_all'] = metric.AverageMeter('loss all rank')
        eval_indices = None
        if hasattr(self, 'dataset'):
            target_ds = self.dataset.get('val') or self.dataset.get('test')
            eval_indices = getattr(target_ds, 'eval_indices', None) if target_ds is not None else None
        self.meter['ap'] = metric.AveragePrecisionMeter(difficult_examples=False, eval_indices=eval_indices)

    @staticmethod
    def convertDict_state(cpk):
        import collections
        cpk_ = collections.OrderedDict()
        for k, v in cpk.items():
            if k.startswith('module.'):
                cpk_[k[7:]] = v
        return cpk_
