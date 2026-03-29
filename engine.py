import torch
import torch.nn as nn
import time, os, shutil
from torch.cuda.amp import GradScaler, autocast
from utilities_s2 import utils, metric, utils_ddp, warmup, logger
from SpliceMix import SpliceMix
import models_s2 as models
from models_s2.SpliceMix_CL import Loss_fn
# ================= 新增：引入 CAM 模块 =================
from models_s2.cam_splicemix import CAMSpliceMixer_MixedLabel
from models_s2.loss_fns import Loss_fn_CAM
# =======================================================
def _compute_consistency_loss(outputs_origin, outputs_syn, tgt_mask, weight=1.0):
                    """
                    【一致性学习Loss】
                    
                    在干净标签位置上，拼接后的预测应该与原始预测保持一致
                    
                    Args:
                        outputs_origin: (B_n, num_classes) - 原始图像的预测
                        outputs_syn: (B_n, num_classes) - 拼接后图像的预测
                        tgt_mask: (B_n, num_classes) - 干净标签掩码（0/1）
                        weight: 一致性损失权重
                    
                    Returns:
                        consistency_loss: 标量损失
                    """
                    import torch.nn.functional as F
                    
                    # ✨ 只在��净标签位置计算一致性
                    # 对于每个样本，在 tgt_mask=1 的位置上，
                    # 拼接后的预测应该接近原始预测
                    
                    # 方案：MSE 损失
                    diff = outputs_syn - outputs_origin  # [B_n, num_classes]
                    squared_diff = diff ** 2             # [B_n, num_classes]
                    
                    # 应用掩码：只在干净标签位置计算
                    masked_diff = squared_diff * tgt_mask  # [B_n, num_classes]
                    
                    # 求平均
                    loss_consistency = masked_diff.sum() / (tgt_mask.sum() + 1e-8)
                    
                    return loss_consistency * weight
class Engine(object):
    def __init__(self, args):
        super(Engine, self).__init__()
        self.args = args
        self.result = {}
        self.result['train'] = {'epoch': [], 'lr': [], 'loss': []}
        self.result['val'] = {'epoch': [], 'loss': [], 'mAUC': [], 'micro_F1': []}
        # 这里的 val_best 记录 mAUC
        self.result['val_best'] = {'epoch': 0, 'loss': -1., 'mAUC': -1., 'metrics': {}}

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
        self.model = getattr(models, self.args.model).model(self.args.num_classes, args=args).to(self.rank)
        self.optimizer = utils.get_optimizer(self.args, self.model)
        self.loss_fn = getattr(models, self.args.model).Loss_fn().to(self.rank)
        # ================= 新增：初始化双轨制所需组件 =================
        # ✨ 使用版本3混合标签模式
        self.cam_mixer = CAMSpliceMixer_MixedLabel(
            use_new_labels=True,        # 融合借用样本的新病灶
            new_label_weight=1.0        # 完全融合（1.0）或概率融合（0.5）
        )
        self.criterion_cam = Loss_fn_CAM().to(self.rank)
        # =============================================================
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
                # 1. 解析 Batch 数据
                inputs, targets, targets_gt, file_name, is_clean, cam_masks = self.on_start_batch(data)
                self.optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.rank)
                
                clean_mask = is_clean.bool()
                noisy_mask = ~clean_mask
                
                # 准备一个空 tensor，用于组装整个 batch 的 output 传给 on_end_batch 统计指标
                outputs_all = torch.zeros((inputs.size(0), self.args.num_classes), device=self.rank)

                # =========================================================
                # 轨道一：干净样本 (保留原版的 SpliceMix 数据增强逻辑)
                # =========================================================
                if clean_mask.sum() > 0:
                    img_c = inputs[clean_mask]
                    tgt_c = targets[clean_mask]
                    
                    with autocast(enabled=not self.args.disable_amp):
                        args_model = {}
                        # 完美复刻原版的干净样本前向逻辑
                        if 'SpliceMix' in self.args.mixer:
                            img_c, tgt_c, flag = self.mixer(img_c, tgt_c)
                            if self.args.model in ['SpliceMix_CL']: 
                                args_model = {'flag': flag}
                        
                        outputs_c = self.model(img_c, args_model)
                        loss_clean = self.loss_fn(outputs_c, tgt_c)
                        
                        # 解析输出并截取对应原始 batch_size 的部分（适应 Tuple 输出）
                        out_c_flat = outputs_c[0] if isinstance(outputs_c, tuple) else outputs_c
                        out_c_flat = out_c_flat[:clean_mask.sum()]
                        
                    weight_c = clean_mask.sum().float() / inputs.size(0)
                    total_loss += loss_clean * weight_c
                    outputs_all[clean_mask] = out_c_flat.data.float()

                # =========================================================
                # 轨道二：噪声样本 (触发 CAM 拼接和局部对齐约束)
                # =========================================================
                # 修改后的代码（Line 120-155）

                if noisy_mask.sum() > 0:
                    img_n = inputs[noisy_mask]
                    tgt_n = targets[noisy_mask]
                    mask_n = cam_masks[noisy_mask]
                    
                    # ✨ 【新增】获取干净样本池
                    clean_imgs = inputs[clean_mask] if clean_mask.sum() > 0 else None
                    clean_tgts = targets[clean_mask] if clean_mask.sum() > 0 else None
                    clean_masks_data = cam_masks[clean_mask] if clean_mask.sum() > 0 else None
                    
                    # 1. 无梯度前向，拿到【原始噪声图像】的预测（��于一致性对比）
                    with torch.no_grad():
                        with autocast(enabled=not self.args.disable_amp):
                            unwrapped_model = self.model.module if hasattr(self.model, 'module') else self.model
                            # ✨ 【关键】保存原始图像的预测（用于一致性学习）
                            outputs_origin_n = unwrapped_model(img_n)
                            outputs_origin_n = outputs_origin_n[0] if isinstance(outputs_origin_n, tuple) else outputs_origin_n
                    
                    # 2. 物理图块拼接 (CAM Splicemix - V3)
                    X_syn, Y_syn, source_grid = self.cam_mixer(
                        img_n, tgt_n, mask_n,
                        clean_imgs, clean_tgts, clean_masks_data
                    )
                    
                    # 3. 动态计算 Loss
                    with autocast(enabled=not self.args.disable_amp):
                        if self.args.model == 'SpliceMix_CL':
                            # 【SpliceMix-CL 分支】
                            outputs_n_overall = self.model(X_syn)
                            out_n_flat = outputs_n_overall[0] if isinstance(outputs_n_overall, tuple) else outputs_n_overall
                            
                            # ✨ 【Loss 1】基础Loss：拼接后图像与拼接后标签
                            loss_noisy_base = self.loss_fn(outputs_n_overall, Y_syn)
                            
                            # ✨ 【Loss 2】一致性学习Loss：拼接后预测与原始预测在干净标签上的一致性
                            loss_consistency = _compute_consistency_loss(
                                outputs_origin_n,      # 原始图像预测
                                out_n_flat,            # 拼接后图像预测
                                tgt_n                  # 原始干净标签（掩码）
                            )
                            
                            # 【双重 Loss】
                            loss_noisy = loss_noisy_base + loss_consistency
                        else:
                            # 【普通 SpliceMix 分支】：只计算基础 Loss
                            outputs_n_overall = self.model(X_syn)
                            out_n_flat = outputs_n_overall[0] if isinstance(outputs_n_overall, tuple) else outputs_n_overall
                            loss_noisy_base = self.loss_fn(outputs_n_overall, Y_syn)
                            loss_noisy = loss_noisy_base
                    
                    # 4. 梯度加权聚合与输出追踪
                    weight_n = noisy_mask.sum().float() / inputs.size(0)
                    total_loss += loss_noisy * weight_n
                    
                    outputs_all[noisy_mask] = out_n_flat.data
                
                # =========================================================
                # 梯度的统一反向传播与状态更新
                # =========================================================
                if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0:
                    self.scaler.scale(total_loss).backward()
                    if self.args.disable_amp:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                
                if self.args.warmup_epochs > 0 and self.epoch <= self.args.warmup_epochs:
                    self.warmup_scheduler.step()

                # 将分离的 out_c 和 out_n 完美拼接，传递给评测代码
                safe_loss = total_loss.detach()
                self.on_end_batch(outputs_all, targets_gt.data, safe_loss, file_name)
            self.on_end_epoch(is_train=True, result=self.result['train'])
            self.lr_scheduler.step()

            if self.args.evaluate > 0 and ((epoch % self.args.evaluate == 0) or epoch == 1):
                self.evaluate(epoch=epoch)

    def evaluate(self, epoch=0):
        torch.cuda.empty_cache()
        val_loader = self.test_loader

        self.model.eval()
        self.on_start_epoch(epoch)
        
        for i, data in enumerate(val_loader):
            # 验证阶段不触发双轨，老老实实走原来的方法
            inputs, targets, targets_gt, file_name, _, _ = self.on_start_batch(data)
            outputs, loss = self.on_forward(inputs, targets, file_name, is_train=False)
            self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

        self.on_end_epoch(is_train=False, result=self.result['val'], result_best=self.result['val_best'])

    def on_forward(self, inputs, targets, file_name, is_train):
        # 训练过程的 forward 已经在 train() 中展开，这里只剩下验证时的干净前向逻辑
        args_model = {}
        with torch.no_grad():
            with autocast(enabled=not self.args.disable_amp):
                outputs = self.model(inputs, args_model)
                loss = self.loss_fn(outputs, targets)

        outputs = outputs[0][:inputs.shape[0]].data if type(outputs) == tuple else outputs[:inputs.shape[0]].data
        return outputs, loss

    def on_start_batch(self, data):
        inputs = data['image'].to(self.rank)
        targets_gt = data['target']
        file_name = data['name']
        targets = targets_gt.clone().to(self.rank)
        targets[targets == -1] = 0
        
        # ================= 【新增】：以字典安全 get 方式提取双轨标志 =================
        # 这样即使您之后更换验证集或其他没有传这两个字段的 Dataset，也不会报错
        #data.get('is_clean', torch.ones(inputs.size(0), dtype=torch.bool))   data中有key为'is_clean'则返回对应的值，否则返回全 True 的布尔张量（默认全是干净样本）
        is_clean = data.get('is_clean', torch.ones(inputs.size(0), dtype=torch.bool)).to(self.rank)
        cam_masks = data.get('cam_mask', torch.ones((inputs.size(0), 2, 2), dtype=torch.bool)).to(self.rank)
        # ============================================================================
        return inputs, targets, targets_gt, file_name, is_clean, cam_masks
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

    def on_end_epoch(self, is_train, result, result_best=None):
        self.lr_curr = utils.get_learning_rate(self.optimizer)
        self.epoch_time = time.time() - self.epoch_time
        meter = self.meter
        loss = meter['loss'].average()
        
        metrics_res = {}

        if utils_ddp.is_main_process():
            loss_all = meter['loss_all'].average()
            
            # --- 核心修改：使用 compute_all_metrics ---
            if not is_train:
                # 只有验证时计算
                metrics_res = meter['ap'].compute_all_metrics()
            else:
                metrics_res = {} # 训练时不计算
        else:
            loss_all = torch.tensor(-1)
            metrics_res = {}

        if self.args.distributed:
            utils_ddp.barrier()

        result['epoch'].append(self.epoch)
        result['loss'].append(loss_all.item())
        if 'mAUC' in metrics_res:
            result['mAUC'].append(metrics_res['mAUC'])

        is_best = False
        
        # --- 格式化日志字符串 (新格式) ---
        str_metrics = ""
        if not is_train and 'mAUC' in metrics_res:
            str_metrics = (
                f"mAUC: {metrics_res['mAUC']:.4f}, "
                f"miF1: {metrics_res['micro_F1']:.4f}, maF1: {metrics_res['macro_F1']:.4f}, "
                f"miP: {metrics_res['micro_P']:.4f}, maP: {metrics_res['macro_P']:.4f}, "
                f"miR: {metrics_res['micro_R']:.4f}, maR: {metrics_res['macro_R']:.4f}"
            )
            # --- 新增：提取并打印每个具体疾病类别的 AUROC ---
            if 'auc_list' in metrics_res:
                auc_list = metrics_res['auc_list']
                # 从 dataset 中获取疾病名称列表，做好容错处理
                if hasattr(self.dataset['test'], 'classes'):
                    class_names = self.dataset['test'].classes
                else:
                    class_names = [f"Class_{i}" for i in range(len(auc_list))]
                
                # 将疾病名称与对应的 AUC 拼接成字符串
                per_class_str = ", ".join([
                    f"{name}: {auc:.4f}" if auc != -1.0 else f"{name}: N/A" 
                    for name, auc in zip(class_names, auc_list)
                ])
                # 打印详细的 Per-Class AUC
                self.logger.info(f"[Per-Class AUC] {per_class_str}")

        if is_train:
            str_log = f'[Epoch {self.epoch}, lr{self.lr_curr}] [Train] time:{utils.strftime(self.epoch_time)}s, loss: {loss:.4f} .'
            self.logger.info(str_log)
        else:
            str_log = f'[Test] time: {utils.strftime(self.epoch_time)}s, loss: {loss:.4f}, {str_metrics} .'
            self.logger.info(str_log)

            # Best Model 判定
            current_mAUC = metrics_res.get('mAUC', 0.0)
            if result_best['mAUC'] < current_mAUC:
                is_best = True
                result_best['mAUC'] = current_mAUC
                result_best['epoch'] = self.epoch
                result_best['loss'] = loss
                result_best['metrics'] = metrics_res

            str_best = f"--[Test-best] (E{result_best['epoch']}), mAUC: {result_best['mAUC']:.4f}"
            self.logger.info(str_best)

        if self.args.evaluate != 0 and utils_ddp.is_main_process():
            self.save_checkpoint(is_train, is_best)
            
        if self.args.distributed:
            utils_ddp.barrier()

    def save_checkpoint(self, is_train, is_best):
        opj = os.path.join
        file = f'ChkpotLast_L{self.args.lr:.1e}_{self.args.model}.pt'
        
        if self.args.distributed:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        checkpoint = {
            'epoch': self.epoch,
            'state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'result_best': self.result['val_best'],
            'args': self.args,
        }
        
        if is_best:
            method_name = "baseline"
            if 'SpliceMix' in self.args.mixer:
                method_name = 'splicemix'
            
            file_best = f'{self.args.data_set}_{method_name}_best.pt'
            file_best_path = opj(self.args.save_path, file_best)
            torch.save(checkpoint, file_best_path)
            self.logger.info(f"Saved best model to {file_best_path}")

    def load_checkpoint(self):
        if self.args.resume == '':
            return
        else:
            file = self.args.resume
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            # 加载文件
            checkpoint = torch.load(file, map_location=map_location)
        
        # 尝试恢复 Log 信息
        try:
            if self.args.start_epoch == 0:
                self.args.start_epoch = checkpoint['epoch'] + 1
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