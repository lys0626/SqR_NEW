import torch
import torchvision
import torch.nn as nn
import models_s2.loss_fns as loss_fns

class model(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(model, self).__init__()
        M = torchvision.models.resnet50(pretrained=pretrained)
        # --- 核心修改：拆分 Backbone ---
        # Stage 1: 提取浅层特征 (至 layer2，特征图尺寸适中，适合拼接)
        self.stage1 = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool, M.layer1, M.layer2,M.layer3)
        # Stage 2: 提取深层特征
        self.stage2 = nn.Sequential( M.layer4)

        # self.stage1 = nn.Sequential(M.conv1, M.bn1, M.relu, M.maxpool, M.layer1)
        # # # Stage 2: 提取深层特征
        # self.stage2 = nn.Sequential(M.layer2, M.layer3, M.layer4)

        self.num_classes = num_classes
        self.glb_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.cls = nn.Linear(M.layer4[-1].conv3.out_channels, num_classes)

    def forward(self, inputs, args=None):  #

        feas = self.stage1(inputs)  
        
        # 如果是验证模式，或者没有传入 mixer，则常规前向传播
        if not self.training or args is None or 'mixer' not in args:
            fea_gmp = self.glb_pooling(self.stage2(feas)).flatten(1)  
            return self.cls(fea_gmp)

        # 2. 特征层拼接 (Feature-level Splicing)
        mixer = args['mixer']
        targets_orig = args['targets']
        
        # 生成拼接后的特征和新的混合标签
        feas_mix, targets_mix, flag = mixer(feas, targets_orig)

        # 3. 拼接特征通过深层网络
        fea_gmp = self.glb_pooling(self.stage2(feas_mix)).flatten(1)  
        output_mix = self.cls(fea_gmp)    

        # 训练时返回混合预测和混合标签，交给 Loss 计算
        return (output_mix, targets_mix)

    def get_config_optim(self, lr, lrp):
        # 适配拆分后的 stage1 和 stage2 参数
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
        
    def forward(self, inputs, targets_ignored):
        # --- 核心修改：兼容特征拼接返回的 Tuple ---
        if isinstance(inputs, tuple) and len(inputs) == 2:
            preds_mix, targets_mix = inputs
            # 使用 SpliceMix 生成的混合 targets_mix 计算 Loss
            return self.loss_fn(preds_mix, targets_mix)
        else:
            # 兼容常规验证模式
            return self.loss_fn(inputs, targets_ignored)

if __name__ == '__main__':
    inputs = torch.randn((2, 3, 448, 448)).cuda()
    target = torch.zeros((2, 20)).cuda()
    target[:, 1:3] = 1

    loss_fn = Loss_fn()

    model = model(20).cuda()
    output = model(inputs)

    loss = loss_fn(output, target)
    loss.backward()

    a= 'pause'
