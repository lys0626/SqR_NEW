import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models_s2.loss_fns as loss_fns

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        M = torchvision.models.resnet50(pretrained=pretrained)
        
        # --- 1. 拆分 Backbone ---
        # Stage 1: 提取浅层特征 (用于拼接操作)
        self.stage1 = nn.Sequential(
            M.conv1, M.bn1, M.relu, M.maxpool,
            M.layer1, M.layer2  # 在 layer2 输出端 (512通道) 截断进行特征拼接
        )
        
        # Stage 2: 提取深层特征
        self.stage2 = nn.Sequential(
             M.layer3, M.layer4
        )
        
        self.num_classes = num_classes
        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

    def forward(self, inputs, args=None):
        # 1. 提取浅层特征
        feas = self.stage1(inputs)
        
        # 测试阶段：直接过完剩余网络
        if not self.training or args is None or 'mixer' not in args:
            feas = self.stage2(feas)
            fea_gp = self.glb_pooling(feas).flatten(1)
            return self.cls(fea_gp)
            
        # --- 训练阶段：特征级 SpliceMix-CL (空间约束版) ---
        mixer = args['mixer']
        targets_orig = args['targets']
        
        # 2. 在浅层进行特征拼接
        feas_all, targets_all, flag = mixer(feas, targets_orig)
        
        # 3. 将所有特征统一通过 Stage 2 进行深度融合
        feas_out = self.stage2(feas_all)
        
        # 获取全局预测 (用于主 BCE Loss)
        preds_all = self.cls(self.glb_pooling(feas_out).flatten(1))
        
        if 'mix_dict' not in flag:
            return (preds_all, targets_all)
            
        # 4. 解析结果
        mix_ind = flag['mix_ind']
        mix_dict = flag['mix_dict']
        
        # 提取深层特征图 (用于切碎) 和 原图预测 (用于教师目标)
        feas_m = feas_out[mix_ind.bool()]
        preds_r = preds_all[(1 - mix_ind).bool()]
        bs_m, C, h, w = feas_m.shape
        
        ng_list = []
        preds_m = preds_m_r = torch.tensor([], device=inputs.device)
        
        # =====================================================================
        # 5. 完美复刻原版 SpliceMix 的空间切分逻辑 (Spatial-Preserving CL)
        # =====================================================================
        for i, (rand_ind, g_row, g_col, n_drop, drop_ind) in enumerate(zip(
            mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], 
            mix_dict['n_drops'], mix_dict['drop_inds'])):
            
            ng = len(rand_ind) // (g_row * g_col)
            fea_m = feas_m[sum(ng_list): sum(ng_list) + ng]
            ng_list.append(ng)
            
            # 如果特征图大小不能被网格整除，进行插值对齐
            if h % g_row + w % g_col != 0:
                fea_m = F.interpolate(fea_m, (h // g_row * g_row, w // g_col * g_col), mode='bilinear', align_corners=True)
                
            # 切分深层特征图 (Chunks)
            chunks = [c.split(fea_m.shape[-1] // g_col, dim=-1) for c in fea_m.split(fea_m.shape[-2] // g_row, dim=-2)]
            fea_m = torch.stack([torch.stack(c, dim=1) for c in chunks], dim=1)  # ng, g_row, g_col, C, h', w'
            fea_m = fea_m.view(-1, C, fea_m.shape[-2], fea_m.shape[-1])  # 展平为独立的小 block
            
            # 教师目标：原图在对应位置的预测
            pred_m_r = torch.masked_fill(preds_r[rand_ind], drop_ind[:, None]==1, -1e3)
            
            # 学生预测：将切碎后的深层特征块单独过池化和分类器
            fea_m_gp = self.glb_pooling(fea_m).flatten(1)
            pred_m = self.cls(fea_m_gp)
            pred_m = torch.masked_fill(pred_m, drop_ind[:, None]==1, -1e3)
            
            # 拼接
            preds_m = torch.cat((preds_m, pred_m), dim=0)
            preds_m_r = torch.cat((preds_m_r, pred_m_r), dim=0)
        # =====================================================================

        # 返回四元组：全局预测、碎块预测、原图预测目标、全局目标
        return (preds_all, preds_m, preds_m_r, targets_all)

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.stage1.parameters())) + list(map(id, self.stage2.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.stage1.parameters(), 'lr': lr * lrp},
            {'params': self.stage2.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

class Loss_fn(loss_fns.BCELoss):
    def __init__(self):
        super(Loss_fn, self).__init__()
        self.bce = self.loss_fn

    def forward(self, inputs, targets_ignored):
        if isinstance(inputs, tuple) and len(inputs) == 4:
            # preds_m 是切碎后的预测, preds_m_r 是原图教师预测
            preds_all, preds_m, preds_m_r, targets_all = inputs
            
            # 1. 主分类损失：作用于完整的图像（包括原始图和混合图的全局预测）
            loss_bce = self.bce(preds_all, targets_all)
            
            # 2. 一致性损失：作用于局部网格（迫使深层特征依然保持空间定位能力）
            # 统一使用 detach() 阻断梯度回传给 Teacher
            loss_cl = self.bce(preds_m, preds_m_r.sigmoid().detach())
            
            return loss_bce + loss_cl
            
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            preds_all, targets_all = inputs
            return self.bce(preds_all, targets_all)
            
        return self.bce(inputs, targets_ignored)
#下面是采用max来计算一致性损失
# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import models_s2.loss_fns as loss_fns

# class model(nn.Module):
#     def __init__(self, num_classes, pretrained=True, args=None):
#         super(model, self).__init__()
#         M = torchvision.models.resnet50(pretrained=pretrained)
        
#         # --- 1. 拆分 Backbone ---
#         # Stage 1: 提取浅层特征 (用于拼接操作)
#         self.stage1 = nn.Sequential(
#             M.conv1, M.bn1, M.relu, M.maxpool,
#             M.layer1, M.layer2  # 在 layer2 输出端 (512通道) 截断进行特征拼接
#         )
        
#         # Stage 2: 提取深层特征
#         self.stage2 = nn.Sequential(
#              M.layer3,M.layer4
#         )
        
#         self.num_classes = num_classes
#         self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
#         self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

#     def forward(self, inputs, args=None):
#         # 1. 提取浅层特征
#         feas = self.stage1(inputs)
        
#         # 测试阶段：直接过完剩余网络
#         if not self.training or args is None or 'mixer' not in args:
#             feas = self.stage2(feas)
#             fea_gp = self.glb_pooling(feas).flatten(1)
#             return self.cls(fea_gp)
            
#         # --- 训练阶段：特征级 SpliceMix-CL ---
#         mixer = args['mixer']
#         targets_orig = args['targets']
        
#         # 2. 在特征层进行拼接
#         # 此时的 feas_all 已经包含了 [原特征, 混合特征]，targets_all 同理
#         feas_all, targets_all, flag = mixer(feas, targets_orig)
        
#         # 3. 将所有特征统一通过 Stage 2 和分类器
#         feas_out = self.stage2(feas_all)
#         preds_all = self.cls(self.glb_pooling(feas_out).flatten(1))
        
#         # 如果没有触发 mix (比如 random probability 没中)
#         if 'mix_dict' not in flag:
#             return (preds_all, targets_all)
            
#         # 4. 解析结果，准备计算 Consistency Loss
#         mix_ind = flag['mix_ind']
#         mix_dict = flag['mix_dict']
        
#         # 分离原图预测和混合图预测
#         preds_orig = preds_all[(1 - mix_ind).bool()]
#         preds_mix = preds_all[mix_ind.bool()]
        
#         # 5. 构建 Consistency Learning 的“教师”预测 (preds_m_r)
#         # 混合特征图的预测，应当与其来源特征的综合预测对齐
#         preds_m_r = torch.tensor([], device=inputs.device)
#         for rand_ind, g_row, g_col, n_drop, drop_ind in zip(
#             mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], 
#             mix_dict['n_drops'], mix_dict['drop_inds']):
            
#             ng = len(rand_ind) // (g_row * g_col)
#             # 取出用于拼接的原图预测
#             pred_orig_teacher = preds_orig[rand_ind]
            
#             # mask 掉被 drop 的区域
#             if n_drop > 0:
#                 pred_orig_teacher = torch.masked_fill(pred_orig_teacher, drop_ind[:, None]==1, -1e3)
            
#             # 聚合预测结果：同一张混合图来源于 g_row * g_col 张原图
#             # 取这几张原图预测的 Max，作为对齐目标
#             pred_teacher_aggregated = pred_orig_teacher.view(ng, g_row * g_col, -1).max(dim=1)[0]
#             preds_m_r = torch.cat((preds_m_r, pred_teacher_aggregated), dim=0)

#         # 返回 Tuple：包含预测结果、对齐目标、以及合并后的真实标签 (交给 Loss_fn 解析)
#         return (preds_all, preds_mix, preds_m_r, targets_all)

#     def get_config_optim(self, lr, lrp):
#         small_lr_layers = list(map(id, self.stage1.parameters())) + list(map(id, self.stage2.parameters()))
#         large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
#         return [
#             {'params': self.stage1.parameters(), 'lr': lr * lrp},
#             {'params': self.stage2.parameters(), 'lr': lr * lrp},
#             {'params': large_lr_layers, 'lr': lr},
#         ]

# class Loss_fn(loss_fns.BCELoss):
#     def __init__(self):
#         super(Loss_fn, self).__init__()
#         self.bce = self.loss_fn

#     def forward(self, inputs, targets_ignored):
#         # 如果启用了 SpliceMix，inputs 包含了我们在 forward 里打包的 4 个变量
#         if isinstance(inputs, tuple) and len(inputs) == 4:
#             preds_all, preds_mix, preds_m_r, targets_all = inputs
            
#             # 1. 主分类损失：直接对包括原图和特征拼接图在内的所有数据计算 BCE
#             loss_bce = self.bce(preds_all, targets_all)
            
#             # 2. 一致性损失 (CL)：混合特征的预测，应当与原图预测的聚合结果对齐
#             # detach() 阻断梯度传回原图，保证训练稳定
#             loss_cl = self.bce(preds_mix, preds_m_r.sigmoid().detach())
            
#             return loss_bce + loss_cl
            
#         elif isinstance(inputs, tuple) and len(inputs) == 2:
#             preds_all, targets_all = inputs
#             return self.bce(preds_all, targets_all)
            
#         # 测试阶段回退到默认计算
#         return self.bce(inputs, targets_ignored)
