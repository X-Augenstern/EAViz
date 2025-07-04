from torch.nn import Module, ModuleList, Conv1d, MultiheadAttention, Sequential, Linear, ReLU, Dropout, LayerNorm, \
    Sigmoid
from copy import deepcopy


def clones(module, N):
    """生成N个相同的层。"""
    return ModuleList([deepcopy(module) for _ in range(N)])


class CausalConv1d(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        """
        初始化CausalConv1d层。
        
        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int, optional): 步长，默认为1。
            dilation (int, optional): 膨胀率，默认为1。
            groups (int, optional): 分组卷积数，默认为1。
            bias (bool, optional): 是否使用偏置，默认为True。
        """
        # 计算因果卷积的填充大小
        self.__padding = (kernel_size - 1) * dilation
        # 调用父类的构造函数
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        """
        执行前向传播。
        
        参数:
            input (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 经过因果卷积后的张量。
        """
        # 调用父类的forward方法执行卷积操作
        result = super(CausalConv1d, self).forward(input)
        # 如果有填充，移除填充部分的数据
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class EncoderLayer(Module):
    def __init__(self, size, h, d_ff, cnn_size, dropout):
        """
        初始化编码层。
        
        参数:
            size (int): 输入维度。
            h (int): 注意力头数。
            d_ff (int): 前馈神经网络的隐藏层维度。
            cnn_size (int): 卷积层通道数。
            dropout (float): Dropout概率。
        """
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiheadAttention(size, h, dropout=dropout)
        self.feed_forward = Sequential(
            Linear(size, d_ff),
            ReLU(),
            Dropout(dropout),
            Linear(d_ff, size)
        )
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.conv = CausalConv1d(cnn_size, cnn_size, kernel_size=7, stride=1)

    def forward(self, x):
        """
        执行前向传播。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 编码后的张量。
        """
        query = self.conv(x)
        x = x + self.dropout1(self.self_attn(self.norm1(query), self.norm1(x), self.norm1(x))[0])
        return x + self.dropout2(self.feed_forward(self.norm2(x)))


class TransformerEncoder(Module):
    def __init__(self, layer, N):
        """
        初始化Transformer编码器。
        
        参数:
            layer (nn.Module): 编码层。
            N (int): 编码层的数量。
        """
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        """
        执行前向传播。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 经过编码器后的张量。
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AutoencoderLayer(Module):
    def __init__(self, input_dim, latent_dim):
        """
        初始化自编码器层。
        
        参数:
            input_dim (int): 输入维度。
            latent_dim (int): 隐变量维度（编码后的维度）。
        """
        super(AutoencoderLayer, self).__init__()
        self.encoder = Sequential(
            Linear(input_dim, latent_dim),
            ReLU()
        )
        self.decoder = Sequential(
            Linear(latent_dim, input_dim),
            Sigmoid()  # Sigmoid激活函数将输出限制在[0, 1]范围内，适用于输入特征的标准化。
        )

    def forward(self, x):
        """
        执行前向传播。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 解码后的张量。
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TransformerAEClassifier(Module):
    def __init__(self):
        """
        初始化Transformer + AE分类器。
        """
        super(TransformerAEClassifier, self).__init__()
        N = 3
        cnn_size = 32
        h = 7
        d_model = 98
        d_ff = 64
        dropout = 0.1
        num_classes = 2

        # 定义Transformer编码器部分
        layer = EncoderLayer(d_model, h, d_ff, cnn_size, dropout)
        self.encoder = TransformerEncoder(layer, N)

        # 定义自编码器部分
        self.autoencoder = AutoencoderLayer(d_model * cnn_size, 64)

        # 定义分类器部分
        self.fc = Sequential(
            Linear(d_model * cnn_size, 64),
            ReLU(),
            Linear(64, num_classes)
        )

    def forward(self, x):
        """
        执行前向传播。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 经过分类器后的张量。
        """
        # 编码输入数据
        encoded_features = self.encoder(x).view(x.size(0), -1)
        # 使用自编码器进行特征压缩和解压缩
        reconstructed_features = self.autoencoder(encoded_features)
        # 通过解码后的特征进行分类
        output = self.fc(reconstructed_features)
        return output

# if __name__ == "__main__":
#     # 检测是否有可用的GPU设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 创建一个随机输入张量
#     input_data = torch.randn(32, 32, 98).to(device)
#     # 初始化模型并将其移动到指定设备上
#     model = TransformerAEClassifier().to(device)
#     # 通过模型获取输出
#     output = model(input_data)
#     # 打印输出的尺寸
#     print(output.size())
