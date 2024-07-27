from torch import cat, matmul, ones, zeros
import torch.nn as nn
from torch.nn.functional import softmax, relu
from math import sqrt
from copy import deepcopy
from sys import path as sys_path
from os import path as os_path

sys_path.append(os_path.abspath('./'))  # 添加上一级目录到sys.path


# SE层中包含乘法的部分，即一个自适应池化和两个全连接
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # nn.AdaptiveAvgPool1d(1) 是一个 PyTorch 中的自适应平均池化层，它的作用是对输入进行全局平均池化，并且输出的形状为 (batch_size, num_channels, 1)。
        # 具体来说，假设输入张量的形状为 (N, C, L)，其中 N 表示 batch size，C 表示通道数，L 表示序列长度。使用 nn.AdaptiveAvgPool1d(1) 后，将对输入在序列长度这个维度上进行全局平均池化，最终输出形状为 (N, C, 1)。
        # 这种操作能够保留每个通道的特征的整体统计信息，有助于提取更加全局的特征表示，通常用于处理时序数据或者具有时间序列特性的数据。

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # 获取输入张量 x 的形状信息，其中 b 表示 batch size，c 表示通道数
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)  # 将y的形状进行扩展，使得和x的形状相同，再相乘 
    # SE模块完成


# SE层中包含卷积和残差的部分
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample  # 下采样层，用于调整输入和输出通道不一致的情况
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=14, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=2, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 128, kernel_size=2, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128

        # 残差连接层
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)  # afr_reduced_cnn_size：经过CNN卷积后的通道数

    # 建立一个包含残差连接的层，确保在输入的数据大小和维度变化时，能够进行适当的残差连接
    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        # 将第一个残差块添加到layers列表中
        layers = [block(self.inplanes, planes, stride, downsample)]

        # 每次创建一个残差块前，都会更新self.inplanes的值，以确保self.inplanes 和 planes始终保持对应关系
        self.inplanes = planes * block.expansion

        # 每次循环都会调用SEBasicBlock，并将当前的self.inplanes 和 planes 作为参数传递
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)  # 将x输入到CNN网络的分支1中
        x2 = self.features2(x)
        # 分别对x1和x2进行z-score标准化  
        x1_mean = x1.mean(dim=0, keepdim=True)  # 沿着batch_size维度计算均值  
        x1_std = x1.std(dim=0, keepdim=True)  # 沿着batch_size维度计算标准差  
        x1_normalized = (x1 - x1_mean) / (x1_std + 1e-5)  # 应用z-score标准化  

        x2_mean = x2.mean(dim=0, keepdim=True)  # 沿着batch_size维度计算均值  
        x2_std = x2.std(dim=0, keepdim=True)  # 沿着batch_size维度计算标准差  
        x2_normalized = (x2 - x2_mean) / (x2_std + 1e-5)  # 应用z-score标准化  

        # 假设你想要在标准化后拼接x1和x2，它们需要有相同的channels数  
        # 如果channels数不同，你需要调整它们（例如，通过添加全连接层改变channels数）  
        # 这里假设channels数相同，直接拼接  
        x_concat = cat((x1_normalized, x2_normalized), dim=2)

        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)

        return x_concat


##########################################################################################

def attention(query, key, value, dropout=None):
    """Implementation of Scaled dot product attention"""
    d_k = query.size(-1)  # 获取查询的维度
    scores = matmul(query, key.transpose(-2, -1)) / sqrt(d_k)  # 计算注意力分数

    p_attn = softmax(scores, dim=-1)  # 使用softmax函数计算注意力权重
    if dropout is not None:
        p_attn = dropout(p_attn)  # 如果指定了dropout，则应用dropout处理注意力权重

    output = matmul(p_attn, value)  # 使用注意力权重加权值
    return output, p_attn  # 返回加权后的输出和注意力权重


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        # 最主要的差别就在于这个填充方式
        self.__padding = (kernel_size - 1) * dilation  # 定义了用于因果卷积的填充方式

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,  # 将定义的填充方式传递给父类的初始化方法
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)  # 调用了父类的 forward 方法执行卷积操作
        if self.__padding != 0:
            return result[:, :, :-self.__padding]  # 检查卷积是否有填充，若存在填充，舍去填充部分数据，去除了输入张量的末尾部分数据
        return result


# 拷贝和输入的module相同的网络
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):  # 定义一个名为MultiHeadedAttention的类，继承自nn.Module
    # 多头注意力检测，将query、key和value设置为权重分配的三个参数，先卷积后注意力
    # 每个参数映射多个头进行运算是为了提取到更多的特征
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()  # 调用父类的构造方法
        assert d_model % h == 0  # 断言，确保d_model能被h整除
        self.d_k = d_model // h  # 计算每个头的key的维度
        self.h = h  # 设置头的数量

        # 初始化卷积层
        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size,
                                         kernel_size=7, stride=1), 3)  # 使用clones函数初始化3个相同的卷积层
        self.linear = nn.Linear(d_model, d_model)  # 初始化线性层
        self.dropout = nn.Dropout(p=dropout)  # 初始化dropout层

    def forward(self, query, key, value):
        """Implements Multi-head attention"""
        nbatches = query.size(0)  # 获取batch的大小

        # 对query、key、value进行维度调整和卷积处理
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)  # 调用attention函数计算多头注意力

        # 维度转换和形状调整
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # 对x进行维度转换和形状调整

        return self.linear(x)  # 返回经过线性层处理后的结果


class LayerNorm(nn.Module):
    """Construct a layer normalization module."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 定义两个可学习的缩放参数，初始值分别为1和0
        self.a_2 = nn.Parameter(ones(features))
        self.b_2 = nn.Parameter(zeros(features))
        # 设置一个小常数，防止除数为0
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    # 残差连接层


class SublayerOutput(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class TCE(nn.Module):
    """
    Transformer Encoder

    It is a stack of N layers.
    """

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# transformer编码层，包括自注意力和前馈神经网络；每个编码器都包含这两个子层
# 每个子层都有残差连接和层标准化
class EncoderLayer(nn.Module):
    """
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    """

    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Add and Normalize层
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        """Transformer Encoder"""
        # 卷积操作的生成结果query获取
        query = self.conv(x_in)

        # self_attn应用自注意力机制，sublayer_output[0]调用第一个子层，即Add&LayerNorm
        # 首先的self_attn中的query, x_in, x_in表示注意力机制中的query、key和value，然后将结果赋值给lambda x
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))

        # 将自注意力的输出应用于前馈神经网络，即两个全连接，sublayer_output[1]调用第二个子层，Add&LayerNorm
        return self.sublayer_output[1](x, self.feed_forward)


# 定义前馈网络，包含两个线性层和一个激活函数
class PositionwiseFeedForward(nn.Module):
    """Positioned feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Implements FFN equation."""
        return self.w_2(self.dropout(relu(self.w_1(x))))


class Celestial(nn.Module):
    def __init__(self):
        super(Celestial, self).__init__()

        N = 2
        afr_reduced_cnn_size = 30
        h = 7
        d_model = 77
        d_ff = 64
        dropout = 0.1
        num_classes = 2
        # self.mfc = MRCNN(afr_reduced_cnn_size)
        self.mfc = MRCNN(afr_reduced_cnn_size)

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        x_feature = self.mfc(x)
        encoded_features = self.tce(x_feature)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)

        return final_output

# if __name__ == "__main__":
#     device = device("cuda" if cuda.is_available() else "cpu")
#
#     input_data = randn(32, 1, 100)
#     labels = randint(0, 2, (32,))
#
#     labels_one_hot = one_hot(labels, num_classes=2).float()
#
#     # CEL = torch.nn.CrossEntropyLoss()
#     BCE = nn.BCEWithLogitsLoss(weight=tensor([10, 1]))
#     print(input_data.size())
#     model = Celestial()
#
#     # 调用模型进行前向传播
#     output = model(input_data)
#     print(output.size())
#     print(labels.size())
#     loss = BCE(output, labels_one_hot)
#     print(loss)
