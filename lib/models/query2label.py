# --------------------------------------------------------
# Quert2Label
# Written by Shilong Liu
# --------------------------------------------------------

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import math

from lib.models.backbone import build_backbone
from lib.models.transformer import build_transformer
from lib.utils.misc import clean_state_dict

#构建分类头
class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        #权重W和偏置b前面带1的目的是利用pytorch的广播机制方便后续计算，判断兼容性的规则是从后往前对齐维度，加1的目的一种良好的编程习惯
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        #stdv是transformer隐藏层维度的倒数平方根，用于初始化权重参数
        #uniform是均匀分布初始化方法，防止梯度消失或梯度爆炸
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)#.data的含义是直接访问张量的底层数据，绕过了PyTorch的自动求导机制
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Qeruy2Label(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.                      #主干网络
            transfomer ([type]): transformer model.                 #transformer decoder
            num_class ([type]): number of classes. (80 for MSCOCO). #数据集的类别数量
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class

        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        #获取transformer的隐藏层维度，为了统一整个网络中特征向量的尺寸，确保数据在不同模块间正确传递和计算，transformer模块内部都是在特定的特征维度上进行计算的
        hidden_dim = transfomer.d_model
        #将backbone输出的特征图通道数转换为transformer所需的隐藏层维度
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.fc_splicemix = nn.Linear(2048, self.num_class)
        #num_embeddings指的是词库大小，即该数据集一共有多少个类别
        #embedding_dim特征维度，每个类别对应的特征向量的长度 
        #可学习参数，如果使用别人已经训练好的词向量(Word2Vec, GloVe等)，可以冻结该层的参数
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        #构建分类头
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)


    # 【修改点】：增加了 return_attn=False 参数
    def forward(self, input, return_attn=False):
        # 1. Backbone 前向传播
        outputs = self.backbone(input)
        
        if isinstance(outputs, tuple):
            features, pos = outputs
        else:
            features = outputs
            pos = None

        # 取最后一层特征
        if isinstance(features, list):
            src = features[-1]
            pos = pos[-1]
        else:
            src = features
        
        # 固定尺寸无需掩码
        mask = None 

        # 2. 准备 Transformer 输入
        query_embed = self.query_embed.weight
        src_proj = self.input_proj(src)

        # 3. --- 路径 A: Transformer 分支 ---
        # 如果 pos 为 None (例如 Stage 2 换上了标准 CNN Backbone)，则直接跳过 Transformer 分支计算以节省显存和算力
        if pos is not None:
            # 【修改点】：接收 transformer 传出来的 attn_weights
            hs, memory, attn_weights = self.transformer(src_proj, query_embed, pos, mask)
            out_features = hs[-1]
            out_logits = self.fc(out_features)
        else:
            out_features = None
            out_logits = None
            attn_weights = None # 【修改点】

        # 4. --- 路径 B: SpliceMix 分支 --- 
        features_pooled = src.amax(dim=[2, 3])
        out_splicemix = self.fc_splicemix(features_pooled)
        
        # 【修改点】：如果要求返回注意力图，则包含 attn_weights
        if return_attn:
            return out_logits, out_features, out_splicemix, src, attn_weights
            
        # 正常训练默认不返回，节省显存与带宽
        return out_logits, out_features, out_splicemix, src

    #收集除了骨干网络backbone以外的所有可训练参数,以便在优化器中对它们进行单独的配置(通常是设置更高的学习率)
    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())
    
    #加载预训练好的权重到模型的骨干网络部分
    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        #map_location参数用于指定加载模型时的设备位置，确保在分布式训练环境中每个进程都能正确加载对应的模型权重
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        self.backbone[0].body.load_state_dict(clean_state_dict(checkpoint['state_dict']), strict=False)
        print("=> loaded ch eckpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))


def build_q2l(args):
    backbone = build_backbone(args) #获取一个批次的特征图和位置编码。作为K,V输入到transformer中
    transformer = build_transformer(args)#构建transformer Decoder模块

    model = Qeruy2Label(
        backbone = backbone,
        transfomer = transformer,
        num_class = args.num_class
    )
    #是否要对输入特征图进行投影
    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    
    return model