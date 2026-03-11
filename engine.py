import torch
import torch.nn as nn
import time, os, shutil
from torch.cuda.amp import GradScaler, autocast
from utilities_s2 import utils, metric, utils_ddp, warmup, logger
from SpliceMix import SpliceMix
import models_s2 as models
from models_s2.SpliceMix_CL import Loss_fn
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
            self.mixer = SpliceMix.SpliceMix(mode=self.args.mixer, grids=self.args.grids,
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

    def evaluate(self, epoch=0):
        torch.cuda.empty_cache()
        val_loader = self.test_loader

        self.model.eval()
        self.on_start_epoch(epoch)
        
        for i, data in enumerate(val_loader):
            inputs, targets, targets_gt, file_name = self.on_start_batch(data)
            outputs, loss = self.on_forward(inputs, targets, file_name, is_train=False)
            self.on_end_batch(outputs, targets_gt.data, loss.data, file_name)

        self.on_end_epoch(is_train=False, result=self.result['val'], result_best=self.result['val_best'])

    def on_forward(self, inputs, targets, file_name, is_train):
        args = {}
        if is_train:
            with autocast(enabled=not self.args.disable_amp):
                if 'SpliceMix' in self.args.mixer:
                    inputs, targets, flag = self.mixer(inputs, targets)
                if self.args.model in ['SpliceMix_CL']: args = {'flag': flag,}
                outputs = self.model(inputs, args)
                loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.args.disable_amp:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
                f"miP: {metrics_res['micro_P']:.4f}, maP: {metrics_res['macro_P']:.4f}"
            )
        
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