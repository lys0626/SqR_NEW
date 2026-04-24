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
             M.layer3,M.layer4
        )
        # self.stage1 = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool, M.layer1)
        # # # Stage 2: 提取深层特征
        # self.stage2 = nn.Sequential(M.layer2, M.layer3, M.layer4)
        
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
            
        # --- 训练阶段：特征级 SpliceMix-CL ---
        mixer = args['mixer']
        targets_orig = args['targets']
        
        # 2. 在特征层进行拼接
        # 此时的 feas_all 已经包含了 [原特征, 混合特征]，targets_all 同理
        feas_all, targets_all, flag = mixer(feas, targets_orig)
        
        # 3. 将所有特征统一通过 Stage 2 和分类器
        feas_out = self.stage2(feas_all)
        preds_all = self.cls(self.glb_pooling(feas_out).flatten(1))
        
        # 如果没有触发 mix (比如 random probability 没中)
        if 'mix_dict' not in flag:
            return (preds_all, targets_all)
            
        # 4. 解析结果，准备计算 Consistency Loss
        mix_ind = flag['mix_ind']
        mix_dict = flag['mix_dict']
        
        # 分离原图预测和混合图预测
        preds_orig = preds_all[(1 - mix_ind).bool()]
        preds_mix = preds_all[mix_ind.bool()]
        
        # 5. 构建 Consistency Learning 的“教师”预测 (preds_m_r)
        # 混合特征图的预测，应当与其来源特征的综合预测对齐
        preds_m_r = torch.tensor([], device=inputs.device)
        for rand_ind, g_row, g_col, n_drop, drop_ind in zip(
            mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], 
            mix_dict['n_drops'], mix_dict['drop_inds']):
            
            ng = len(rand_ind) // (g_row * g_col)
            # 取出用于拼接的原图预测
            pred_orig_teacher = preds_orig[rand_ind]
            
            # mask 掉被 drop 的区域
            if n_drop > 0:
                pred_orig_teacher = torch.masked_fill(pred_orig_teacher, drop_ind[:, None]==1, -1e3)
            
            # 聚合预测结果：同一张混合图来源于 g_row * g_col 张原图
            # 取这几张原图预测的 Max，作为对齐目标
            pred_teacher_aggregated = pred_orig_teacher.view(ng, g_row * g_col, -1).max(dim=1)[0]
            preds_m_r = torch.cat((preds_m_r, pred_teacher_aggregated), dim=0)

        # 返回 Tuple：包含预测结果、对齐目标、以及合并后的真实标签 (交给 Loss_fn 解析)
        return (preds_all, preds_mix, preds_m_r, targets_all)

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
        # 如果启用了 SpliceMix，inputs 包含了我们在 forward 里打包的 4 个变量
        if isinstance(inputs, tuple) and len(inputs) == 4:
            preds_all, preds_mix, preds_m_r, targets_all = inputs
            
            # 1. 主分类损失：直接对包括原图和特征拼接图在内的所有数据计算 BCE
            loss_bce = self.bce(preds_all, targets_all)
            
            # 2. 一致性损失 (CL)：混合特征的预测，应当与原图预测的聚合结果对齐
            # detach() 阻断梯度传回原图，保证训练稳定
            loss_cl = self.bce(preds_mix, preds_m_r.sigmoid().detach())
            
            return loss_bce + loss_cl
            
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            preds_all, targets_all = inputs
            return self.bce(preds_all, targets_all)
            
        # 测试阶段回退到默认计算
        return self.bce(inputs, targets_ignored)
# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import models_s2.loss_fns as loss_fns

# class model(nn.Module):
#     def __init__(self, num_classes, pretrained=True, args=None):
#         super(model, self).__init__()
#         M = torchvision.models.resnet50(pretrained=pretrained)
#         # res = torchvision.models.resnet101(pretrained=pretrained, norm_layer=lib.FrozenBatchNorm2d)
#         # res = torchvision.models.resnet50(True)
#         self.backbone = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool,
#                                       M.layer1, M.layer2, M.layer3, M.layer4, )
#         self.num_classes = num_classes

#         self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
#         self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

#     def forward(self, inputs, args=None):  #

#         feas = self.backbone(inputs)  # bs, C, 28, 28
#         fea_gp = self.glb_pooling(feas).flatten(1)  # bs, C
#         preds = self.cls(fea_gp)

#         if self.training:  ## split feature maps
#             mix_ind, mix_dict = args['flag']['mix_ind'], args['flag']['mix_dict']
#             feas_r, preds_r = feas[(1 - mix_ind).bool()], preds[(1 - mix_ind).bool()]
#             feas_m, _ = feas[mix_ind.bool()], preds[mix_ind.bool()]
#             bs_m, C, h, w = feas_m.shape

#             ng_list = []
#             preds_m = preds_m_r = torch.tensor([], device=inputs.device)
#             for i, (rand_ind, g_row, g_col, n_drop, drop_ind) in enumerate(zip(mix_dict['rand_inds'], mix_dict['rows'], mix_dict['cols'], mix_dict['n_drops'], mix_dict['drop_inds'])):  # insertion ordered in Dict after python 3.6 -> better code to be done
#                 ng = len(rand_ind) // (g_row * g_col)
#                 fea_m = feas_m[sum(ng_list): sum(ng_list) + ng]
#                 ng_list.append(ng)
#                 # fea_r, tgt_r = feas_r[rand_ind], None
#                 if h % g_row + w % g_col != 0:
#                     fea_m = F.interpolate(fea_m, (h // g_row * g_row, w // g_col * g_col), mode='bilinear', align_corners=True)
#                 chunks = [c.split(fea_m.shape[-1] // g_col, dim=-1) for c in fea_m.split(fea_m.shape[-2] // g_row, dim=-2)]  # [[[]\in{ng, C, h//g_row(h'), w//g_col(w')},[],...](sub-imgs in row 1), [[],[],...](sub-imgs in row 2), ...]
#                 fea_m = torch.stack([torch.stack(c, dim=1) for c in chunks], dim=1)  # ng, g_row, g_col, C, h', w' || stack in cols per row, then stack in rows
#                 fea_m = fea_m.view(-1, C, fea_m.shape[-2], fea_m.shape[-1])  # ng, C, h, w

#                 # pred_m_r = preds_r[rand_ind] * (1 - drop_ind[:, None]) if n_drop > 0 else preds_r[rand_ind]
#                 pred_m_r = torch.masked_fill(preds_r[rand_ind], drop_ind[:, None]==1, -1e3)


#                 fea_m_gp = self.glb_pooling(fea_m).flatten(1)
#                 pred_m = self.cls(fea_m_gp)
#                 pred_m = torch.masked_fill(pred_m, drop_ind[:, None]==1, -1e3)

#                 preds_m = torch.cat((preds_m, pred_m), dim=0)
#                 preds_m_r = torch.cat((preds_m_r, pred_m_r), dim=0)

#             preds = (preds, preds_m, preds_m_r)
#             # return preds[:(1 - mix_ind).sum()], ...
#         return preds


#     def splitting(self, fea, mix_dict):
#         pass

#     def get_config_optim(self, lr, lrp):
#         small_lr_layers = list(map(id, self.backbone.parameters()))
#         large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
#         return [
#             {'params': self.backbone.parameters(), 'lr': lr * lrp},
#             {'params': large_lr_layers, 'lr': lr},
#         ]


# class Loss_fn(loss_fns.BCELoss):
#     def __init__(self):
#         super(Loss_fn, self).__init__()
#         self.bce = self.loss_fn

#     def forward(self, inputs, targets):
#         if len(inputs) == 3:
#             preds, preds_m, preds_m_r = inputs
#             loss_bce = self.bce(preds, targets)
#             if targets.shape[-1] == 20:  # for VOC2007
#                 loss_cl = self.bce(preds_m, preds_m_r.sigmoid())
#             else:  # for MS-COCO and others
#                 loss_cl = self.bce(preds_m, preds_m_r.sigmoid().detach())
#             loss = loss_bce + loss_cl
#         else:
#             loss = self.bce(inputs, targets)
#         return loss

# if __name__ == '__main__':
#     from SpliceMix import SpliceMix

#     bs = 8
#     inputs = torch.randn((bs, 3, 224, 224)).cuda()
#     target = torch.zeros((bs, 20)).cuda()
#     target[:, 1:3] = 1

#     mixer = SpliceMix(mode='cmix', grids=['1x2', '2x3-2'], n_grids=[1, 2]).mixer
#     imgs_mix, tgts_mix, flag = mixer(inputs, target)
#     args = {'flag': flag}

#     loss_fn = Loss_fn()

#     model = model(20).cuda()
#     output = model(imgs_mix, args)

#     loss = loss_fn(output, tgts_mix)
#     loss.backward()

#     a= 'pause'
