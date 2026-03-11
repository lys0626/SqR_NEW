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

from models.backbone import build_backbone
from models.transformer import build_transformer
from utils.misc import clean_state_dict
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
            self.W[0][i].data.uniform_(-stdv, stdv)#.data的含义是直接访问张量的底层数据，绕过了PyTorch的自动求导机制(在修改张量内容时不被Autograd发现)，现在一般使用with torch.no_grad()来实现类似的功能
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


    def forward(self, input):
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
        # hs: [Layers, Batch, Num_Class, Hidden_Dim] -> Transformer Decoder 的隐层输出
        # memory: [Batch, Hidden_Dim, H, W] -> Transformer Encoder 的空间特征
        # hs, memory = self.transformer(src_proj, query_embed, pos, mask)
        # [关键修正 1] 获取最后一层的类别特征
        # out_features = hs[-1]
        # [关键修正 2] 通过分类头得到最终 Logits
        # out_logits = self.fc(out_features)
        # 3. --- 路径 A: Transformer 分支 ---
        # 如果 pos 为 None (例如 Stage 2 换上了标准 CNN Backbone)，则直接跳过 Transformer 分支计算以节省显存和算力
        if pos is not None:
            hs, memory = self.transformer(src_proj, query_embed, pos, mask)
            out_features = hs[-1]
            out_logits = self.fc(out_features)
        else:
            out_features = None
            out_logits = None
        # 4. --- 路径 B: SpliceMix 分支 ---
        # 原代码: features_pooled = src.mean(dim=[2, 3])
        # 原代码: out_splicemix = self.fc_splicemix(features_pooled)
        
        # [修改] 暂时不在这里做池化和FC，而是直接把 src (空间特征) 返回去
        # 或者为了兼容 Stage 2 的非 CL 模式，我们可以保留 Gap 分支，但额外返回 src
        
        features_pooled = src.amax(dim=[2, 3])
        out_splicemix = self.fc_splicemix(features_pooled)
        
        # 返回: Logits, Class_Features, Aux_Logits, Spatial_Features(新增)
        # src 的 shape: [B, C, H, W] (例如 [32, 2048, 14, 14])
        return out_logits, out_features, out_splicemix, src
    #收集除了骨干网络backbone以外的所有可训练参数,以便在优化器中对它们进行单独的配置(通常是设置更高的学习率)
    ''' 
        self.transformer: Transformer 模块（包括 Encoder 和 Decoder）的所有权重。
        self.fc: 最后的分类头（全连接层）。
        self.input_proj: 将骨干网络特征映射到 Transformer 维度的 1x1 卷积层。
        self.query_embed: 可学习的标签嵌入向量（即 Query）。 如果采用clip的文本编码器或者别的大预言模型可以考虑冻结这一部分(self.query_embed)，
            但是考虑到文本编码器输出的特征维度与transformer的隐藏层维度可能不匹配，一般会添加一个投影层，这个投影层是需要训练的
    '''
    def finetune_paras(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(), self.query_embed.parameters())
    
    #加载预训练好的权重到模型的骨干网络部分
    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))
        #map_location参数用于指定加载模型时的设备位置，确保在分布式训练环境中每个进程都能正确加载对应的模型权重
        checkpoint = torch.load(path, map_location=torch.device(dist.get_rank()))
        # import ipdb; ipdb.set_trace()
        #在Query2Label模型中，self.backbone是一个Joiner对象(Joiner对象指的是将Backone和PositionEmbedding结合在一起的模块)
        #self.backbone[0]是骨干网络Backbone,self.backbone[0].body是Backbone的主体网络部分
        #self.backbone[1]指的是位置编码模块
        #clean_state_dict(checkpoint['state_dict'])，遍历字典中的每一个key，如果发现开头是module.，就去掉这个前缀,解决module前缀不匹配的问题
        #strict=False表示在加载权重时，如果模型的某些层在预训练权重中没有对应的参数，或者预训练权重中有一些额外的参数不在模型中，这些不匹配的情况不会引发错误，解决分类头不匹配的问题
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
    #如果不需要投影(即图像特征的维度与transformer Decoder的hidden layer维度一致时)，则将input_proj设置为Identity层，即不进行任何变换
    if not args.keep_input_proj:
        model.input_proj = nn.Identity()
        print("set model.input_proj to Indentify!")
    

    return model
        
        
