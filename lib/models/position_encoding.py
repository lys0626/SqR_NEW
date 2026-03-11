# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
We borrow the positional encoding from Detr and simplify the model.
"""
import math
import torch
from torch import nn
from torch.functional import Tensor

# from utils.misc import NestedTensor

#借鉴了DETR的实现，将nlp中的一维编码推广到了二维图像上，并进行简化
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    #num_pos_feats:每个轴(x或y)的位置特征维度，最终的位置编码的维度时2*num_pos_feats
    #temperature:控制正弦/余弦函数频率的参数，默认为10000
    #normalize:是否对位置编码进行归一化处理，是否将坐标归一化为[0,1]之间，通常为True,归一化使得位置编码不再依赖于输入图像的绝对尺寸，而是用相对位置,目的是保存位置之间的相对关系
    #scale:归一化后的缩放因子，默认为2π，  先将坐标归一化到[0,1]，再乘以scale得到最终的坐标值[0,2π],目的是防止数值"塌缩"，数值之间保持一定的差异性和适配三角函数的非线性特征
    #maxH,maxW:预定义的最大高度和宽度，用于创建位置编码缓冲区(buffer),在这里面硬编码了，根据不同的输入图像的分辨率计算，结果应该是maxH = args.img_size // downsample_ratio，即输入的图像除以下采样比例
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        ## 生成位置编码的 buffer（缓存），这样在前向传播时不需要重复计算，只需读取
        pe = self._gen_pos_buffer()
        # register_buffer 将变量注册为模型的一部分，但不是可学习的参数（不会随梯度更新）
        self.register_buffer('pe', pe)

    #生成位置编码矩阵
    def _gen_pos_buffer(self):
        #创建一个形状为(1, maxH, maxW)的全1张量，作为累加的基础
        _eyes = torch.ones((1, self.maxH, self.maxW))
        #在第一维，行累加，表示y轴位置,cumsum是累加函数
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        #在第二维，列累加，表示x轴位置
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        #归一化，将坐标归一化到[0,2π]，目的是让位置编码对不同分辨率的图像具有更好的鲁棒性
        if self.normalize:
            eps = 1e-6
            #除以当前维度的最大值进行归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        #计算频率项,目的是具备区分精细位置的能力
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        #生成x,y的位置编码
        #增加维度None是为了利用广播机制
        #进行维度对齐时(从右向左开始对齐)，检查两个维度是否兼容只有两个条件:1.两个维度相等 2.其中一个维度为1
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        #交替使用sin和cos函数编码位置，[sin(x), cos(x), sin(x), cos(x)...]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #拼接x和y的位置编码，得到最终的位置编码张量
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    #前向传播
    #input:形状为(B, C, H, W)的输入张量
    #return:形状为(B, 2*num_pos_feats, H, W)的位置编码张量
    def forward(self, input: Tensor):
        x = input
        ## 将预计算的 buffer 扩展到与输入相同的 Batch Size,repeat是张量复制函数，括号里的数字代表在某个维度上的复制次数
        return self.pe.repeat((x.size(0),1,1,1))

#创建位置编码模块
def build_position_encoding(args):
    # args.hidden_dim 是 Transformer 的总隐藏层维度 (例如 256 或 512)。
    # 因为我们是对 2D 图像进行编码，需要同时编码 X 轴和 Y 轴。
    # 通常的做法是将总维度对半分：一半用于 Y 轴，一半用于 X 轴。
    # 所以 N_steps = hidden_dim / 2。
    N_steps = args.hidden_dim // 2
    #根据backbone的不同，设置不同的下采样比例
    # 如果使用的是 CvT_w24 (Convolutional Vision Transformer) 作为 backbone
    if args.backbone in ['CvT_w24'] :
        downsample_ratio = 16
    else:
        downsample_ratio = 32   #resnet系列的backbone下采样比例都是32
    # 3. 构建位置编码实例
    # 检查请求的位置编码类型。'sine' 和 'v2' 这里都指向正弦位置编码，不论设置哪个效果都一样
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        assert args.img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(args.img_size)
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=args.img_size // downsample_ratio, maxW=args.img_size // downsample_ratio)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    # 1. 模拟配置参数
    class Args:
        hidden_dim = 256
        backbone = 'resnet50'
        position_embedding = 'sine'
        img_size = 224

    args = Args()

    # 2. 构建位置编码实例
    pos_encoder = build_position_encoding(args)
    
    # 3. 打印基础信息
    print("=== Position Encoding Debug Information ===")
    print(f"Buffer shape (pe): {pos_encoder.pe.shape}") 
    # 预期输出: [1, 256, 7, 7] (因为 224/32 = 7)
    
    # 4. 模拟前向传播
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 7, 7)
    pos_output = pos_encoder(dummy_input)
    print(f"Output shape: {pos_output.shape}")
    # 预期输出: [2, 256, 7, 7]

    # 5. 验证归一化效果 (Check if values are within [-1, 1])
    print(f"Value range: min={pos_output.min().item():.4f}, max={pos_output.max().item():.4f}")

    # 6. 可视化
    # 提取前 128 维 (Y 轴编码) 和 后 128 维 (X 轴编码)
    pe_map = pos_encoder.pe.squeeze(0).cpu() # [256, H, W]
    
    plt.figure(figsize=(12, 5))
    
    # 可视化 Y 轴的一个通道 (比如第 0 个通道)
    # 它应该随高度变化（纵向有梯度）
    plt.subplot(1, 2, 1)
    plt.title("Y-axis Encoding (Channel 0)")
    plt.imshow(pe_map[0].numpy(), cmap='viridis')
    plt.colorbar()

    # 可视化 X 轴的一个通道 (比如第 128 个通道)
    # 它应该随宽度变化（横向有梯度）
    plt.subplot(1, 2, 2)
    plt.title("X-axis Encoding (Channel 128)")
    plt.imshow(pe_map[128].numpy(), cmap='viridis')
    plt.colorbar()

    plt.show()
    print("===========================================")