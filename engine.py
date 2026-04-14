import torch
import torch.nn as nn
import time, os, shutil
from torch.cuda.amp import GradScaler, autocast
from utilities_s2 import utils, metric, utils_ddp, warmup, logger
from SpliceMix import SpliceMix
import models_s2 as models
import copy, math
from models_s2.SpliceMix_CL import Loss_fn
import numpy as np
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
        train_set, test_set, self.args.num_classes = utils.get_dataset(self.args)
        self.dataset = {'train': train_set, 'test': test_set}
        self.scaler = GradScaler(enabled=not self.args.disable_amp)

        args = {}
        #getattr动态获取模块
        self.model = getattr(models, self.args.model).model(self.args.num_classes, args=args).to(self.rank)
        self.ema = ModelEMA(self.model)
        self.optimizer = utils.get_optimizer(self.args, self.model)
        self.loss_fn = getattr(models, self.args.model).Loss_fn().to(self.rank)
        
        self.train_loader, self.test_loader = utils.get_dataloader(train_set=self.dataset['train'],
                                                       test_set=self.dataset['test'], args=self.args)
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
                inputs, targets, targets_gt, file_name = self.on_start_batch(data)
                outputs, loss = self.on_forward(inputs, targets, file_name, is_train=True)
                self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

            self.on_end_epoch(is_train=True, result=self.result['train'])
            self.lr_scheduler.step()

            if self.args.evaluate > 0 and ((epoch % self.args.evaluate == 0) or epoch == 1):
                self.evaluate(epoch=epoch)

    # def evaluate(self, epoch=0):
    #     torch.cuda.empty_cache()
    #     val_loader = self.test_loader

    #     self.model.eval()
    #     self.on_start_epoch(epoch)
        
    #     for i, data in enumerate(val_loader):
    #         inputs, targets, targets_gt, file_name = self.on_start_batch(data)
    #         outputs, loss = self.on_forward(inputs, targets, file_name, is_train=False)
    #         self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

    #     self.on_end_epoch(is_train=False, result=self.result['val'], result_best=self.result['val_best'])
    def evaluate(self, epoch=0):
        torch.cuda.empty_cache()
        val_loader = self.test_loader
        
        # 定义需要评估的模型列表
        eval_pairs = []
        # 如果是训练阶段，才需要评估普通模型
        if self.args.is_train:
            # 训练阶段：普通模型和 EMA 模型都要评估以对比记录
            eval_pairs.append({'model': self.model, 'res': self.result['val'], 'best': self.result['val_best'], 'is_ema': False})
            if hasattr(self, 'ema'):
                eval_pairs.append({'model': self.ema.ema, 'res': self.result['ema_val'], 'best': self.result['ema_val_best'], 'is_ema': True})
        else:
            # 纯测试阶段：根据加载的权重类型，只评估对应的一个模型
            is_ema_loaded = getattr(self, 'loaded_is_ema', False)
            
            # 双重保险：如果 checkpoint 字典里写了是 EMA，或者文件名里带了 'EMA'
            if is_ema_loaded or 'EMA' in self.args.resume:
                eval_pairs.append({'model': self.ema.ema, 'res': self.result['ema_val'], 'best': self.result['ema_val_best'], 'is_ema': True})
            else:
                eval_pairs.append({'model': self.model, 'res': self.result['val'], 'best': self.result['val_best'], 'is_ema': False})
        for item in eval_pairs:
            m = item['model']
            m.eval()
            self.on_start_epoch(epoch) # 重置当前模型的统计器
            
            self.logger.info(f"==> Evaluating {'EMA ' if item['is_ema'] else 'Standard '}Model...")
            
            for i, data in enumerate(val_loader):
                inputs, targets, targets_gt, file_name = self.on_start_batch(data)
                
                with torch.no_grad():
                    with autocast(enabled=not self.args.disable_amp):
                        if not item['is_ema']:
                            outputs, loss = self.on_forward(inputs, targets, file_name, is_train=False)
                        else:
                            # EMA 专用前向传播 (无 mixer)
                            outputs = m(inputs)
                            if isinstance(outputs, tuple): outputs = outputs[0]
                            if isinstance(outputs, dict): outputs = outputs['logits_mixed']
                            loss = self.loss_fn(outputs, targets)
                
                outputs = outputs[:inputs.shape[0]].data
                self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

            # 调用修改后的 on_end_epoch 记录结果并保存各自的 Best 权重
            self.on_end_epoch(is_train=False, result=item['res'], result_best=item['best'], is_ema=item['is_ema'])
    def on_forward(self, inputs, targets, file_name, is_train):
        args = {}
        if is_train:
            with autocast(enabled=not self.args.disable_amp):
                # ---------------- 核心修改：移除外部图片拼接 ----------------
                # 原代码： inputs, targets, flag = self.mixer(inputs, targets)
                # 改为将 mixer 和原始 targets 传入模型内部处理
                
                if self.args.model in ['SpliceMix_CL']: 
                    args = {
                        'mixer': self.mixer,
                        'targets': targets
                    }
                
                outputs = self.model(inputs, args)
                # targets_gt 将在 Loss_fn 内部被替换为 targets_all，这里传入原始 targets 仅做占位
                loss = self.loss_fn(outputs, targets)
                # --------------------------------------------------------
                # if 'SpliceMix' in self.args.mixer:
                #     inputs, targets, flag = self.mixer(inputs, targets)
                # if self.args.model in ['SpliceMix_CL']: args = {'flag': flag,}
                # outputs = self.model(inputs, args)
                # loss = self.loss_fn(outputs, targets)

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

        outputs = outputs[0][:inputs.shape[0]].data if type(outputs) == tuple else outputs[:inputs.shape[0]].data
        return outputs, loss

    def on_start_batch(self, data):
        inputs = data['image'].to(self.rank)
        targets_gt = data['target']
        file_name = data['name']
        targets = targets_gt.clone().to(self.rank)
        #将标注为-1的不确定是否存在的标签强制转换为0，视为样本上没有该标签
        targets[targets == -1] = 0
        return inputs, targets, targets_gt, file_name

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
    def on_end_epoch(self, is_train, result, result_best=None, is_ema=False):
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
        log_tag = "Train" if is_train else ("EMA-Test" if is_ema else "Test")
        
        if not is_train and 'mAUC' in metrics_res:
            str_metrics = (
                f"mAUC: {metrics_res['mAUC']:.4f}, "
                f"miF1: {metrics_res['micro_F1']:.4f}, maF1: {metrics_res['macro_F1']:.4f}, "
                f"miR: {metrics_res['micro_R']:.4f}, maR: {metrics_res['macro_R']:.4f}"
            )
            
            # --- 打印每个具体疾病类别的 AUROC (Per-Class) ---
            if 'auc_list' in metrics_res:
                auc_list = metrics_res['auc_list']
                # 尝试从 dataset 中获取疾病名称，增加 EMA 模式下的容错
                target_ds = self.dataset['test']
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
        if not is_train:
            current_mAUC = metrics_res.get('mAUC', 0.0)
            if result_best['mAUC'] < current_mAUC:
                is_best = True
                result_best['mAUC'] = current_mAUC
                result_best['epoch'] = self.epoch
                result_best['loss'] = loss
                result_best['metrics'] = metrics_res

            str_best = f"--[{log_tag}-best] (E{result_best['epoch']}), mAUC: {result_best['mAUC']:.4f}"
            self.logger.info(str_best)

        if not is_train and self.args.evaluate != 0 and utils_ddp.is_main_process():
            self.save_checkpoint(is_best=is_best, is_ema=is_ema) # ✅ 显式传参
        if self.args.distributed:
            utils_ddp.barrier()
    # def on_end_epoch(self, is_train, result, result_best=None,is_ema=False):
    #     self.lr_curr = utils.get_learning_rate(self.optimizer)
    #     self.epoch_time = time.time() - self.epoch_time
    #     meter = self.meter
    #     loss = meter['loss'].average()
        
    #     metrics_res = {}

    #     if utils_ddp.is_main_process():
    #         loss_all = meter['loss_all'].average()
            
    #         # --- 核心修改：使用 compute_all_metrics ---
    #         if not is_train:
    #             # 只有验证时计算
    #             metrics_res = meter['ap'].compute_all_metrics()
    #         else:
    #             metrics_res = {} # 训练时不计算
    #     else:
    #         loss_all = torch.tensor(-1)
    #         metrics_res = {}

    #     if self.args.distributed:
    #         utils_ddp.barrier()

    #     result['epoch'].append(self.epoch)
    #     result['loss'].append(loss_all.item())
    #     if 'mAUC' in metrics_res:
    #         result['mAUC'].append(metrics_res['mAUC'])

    #     is_best = False
        
    #     # --- 格式化日志字符串 (新格式) ---
    #     str_metrics = ""
    #     log_tag = "Train" if is_train else ("EMA-Test" if is_ema else "Test")
    #     if not is_train and 'mAUC' in metrics_res:
    #         str_metrics = (
    #             f"mAUC: {metrics_res['mAUC']:.4f}, "
    #             f"miF1: {metrics_res['micro_F1']:.4f}, maF1: {metrics_res['macro_F1']:.4f}, "
    #             f"miP: {metrics_res['micro_P']:.4f}, maP: {metrics_res['macro_P']:.4f}, "
    #             f"miR: {metrics_res['micro_R']:.4f}, maR: {metrics_res['macro_R']:.4f}"
    #         )
    #         # --- 新增：提取并打印每个具体疾病类别的 AUROC ---
    #         if 'auc_list' in metrics_res:
    #             auc_list = metrics_res['auc_list']
    #             # 从 dataset 中获取疾病名称列表，做好容错处理
    #             if hasattr(self.dataset['test'], 'classes'):
    #                 class_names = self.dataset['test'].classes
    #             else:
    #                 class_names = [f"Class_{i}" for i in range(len(auc_list))]
                
    #             # 将疾病名称与对应的 AUC 拼接成字符串
    #             per_class_str = ", ".join([
    #                 f"{name}: {auc:.4f}" if auc != -1.0 else f"{name}: N/A" 
    #                 for name, auc in zip(class_names, auc_list)
    #             ])
    #             # 打印详细的 Per-Class AUC
    #             self.logger.info(f"[Per-Class AUC] {per_class_str}")

    #     if is_train:
    #         str_log = f'[Epoch {self.epoch}, lr{self.lr_curr}] [Train] time:{utils.strftime(self.epoch_time)}s, loss: {loss:.4f} .'
    #         self.logger.info(str_log)
    #     else:
    #         str_log = f'[Test] time: {utils.strftime(self.epoch_time)}s, loss: {loss:.4f}, {str_metrics} .'
    #         self.logger.info(str_log)

    #         # Best Model 判定
    #         current_mAUC = metrics_res.get('mAUC', 0.0)
    #         if result_best['mAUC'] < current_mAUC:
    #             is_best = True
    #             result_best['mAUC'] = current_mAUC
    #             result_best['epoch'] = self.epoch
    #             result_best['loss'] = loss
    #             result_best['metrics'] = metrics_res

    #         str_best = f"--[Test-best] (E{result_best['epoch']}), mAUC: {result_best['mAUC']:.4f}"
    #         self.logger.info(str_best)

    #     if self.args.evaluate != 0 and utils_ddp.is_main_process():
    #         self.save_checkpoint(is_train, is_best, is_ema=is_ema)
            
    #     if self.args.distributed:
    #         utils_ddp.barrier()

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
            for k, v in loaded_dict.items():
                if k.startswith('module.'):
                    name = k[7:] # 去除 'module.'
                else:
                    name = k
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
        self.meter['ap'] = metric.AveragePrecisionMeter(difficult_examples=False)

    @staticmethod
    def convertDict_state(cpk):
        import collections
        cpk_ = collections.OrderedDict()
        for k, v in cpk.items():
            if k.startswith('module.'):
                cpk_[k[7:]] = v
        return cpk_