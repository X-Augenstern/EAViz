from torch import device, from_numpy, stack, tensor, float32
from torch.nn import Module, AdaptiveAvgPool1d, Sequential, Linear, ReLU, Sigmoid, Conv1d, BatchNorm1d, MaxPool1d
from torch.nn.init import xavier_uniform_, zeros_, ones_
from scipy.signal import welch
from scipy import signal
from numpy import ndarray, finfo, where, log, mean, trapz


class psdWeight(Module):
    def __init__(self, channel, reduction=2):
        """
        初始化 psdWeight 模块。

        参数:
        - channel: 输入通道数。
        - reduction: 通道缩减比例，默认为 2。
        """
        super(psdWeight, self).__init__()
        # 自适应平均池化层，将每个通道的数据池化成一个值
        self.avg_pool = AdaptiveAvgPool1d(1)

        # 全连接层网络序列
        self.fc = Sequential(
            # 第一层全连接层，将通道数从 channel 缩减到 channel // reduction
            Linear(channel, channel // reduction, bias=False),
            ReLU(inplace=True),
            # 第二层全连接层，将通道数恢复到 channel
            Linear(channel // reduction, channel, bias=False),
            # Sigmoid 激活函数，将输出限制在 [0, 1] 之间
            Sigmoid()
        )

    def forward(self, in_psd, in_tim):
        """
        前向传播函数。

        参数:
        - in_psd: 输入的功率谱密度数据，形状为 (batch_size, channels, length)。
        - in_tim: 输入的时间序列数据，形状为 (batch_size, channels, length)。

        返回:
        - 加权后的时间序列数据，形状与 in_tim 相同。
        - 特征权重，形状为 (batch_size, channels, 1)。
        """
        # 获取输入的批量大小和通道数
        b, c, _ = in_psd.size()

        # 自适应平均池化，将每个通道池化成一个值，并重塑为 (batch_size, channels)
        y = self.avg_pool(in_psd).view(b, c)

        # 通过全连接层网络，计算权重，并重塑为 (batch_size, channels, 1)
        y = self.fc(y).view(b, c, 1)

        # 将权重扩展为与 in_tim 形状相同，并对 in_tim 进行加权
        return in_tim * y.expand_as(in_tim), y


class getPsdFeature(Module):
    def __init__(self, fs=1000, nperseg=100, lowFr=80, higFr=450,
                 afr_reduced_cnn_size=32, device=device("cuda")):
        """
        初始化getPsdFeature类。
        
        参数：
        fs (int): 采样频率。
        nperseg (int): 每个段的长度，用于Welch方法。
        lowFr (int): 带通滤波器的低频边界。
        higFr (int): 带通滤波器的高频边界。
        afr_reduced_cnn_size (int): 第二个Conv1d层的输出通道数。
        device (torch.device): 运行模型的设备（CPU或CUDA）。
        """
        super(getPsdFeature, self).__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.lowFr = lowFr
        self.higFr = higFr
        self.device = device

        # 定义卷积神经网络层
        self.features1 = Sequential(
            Conv1d(1, 8, kernel_size=4, stride=1, bias=False, padding=1),
            BatchNorm1d(8),
            ReLU(),
            Conv1d(8, afr_reduced_cnn_size, kernel_size=4, stride=1,
                   bias=False, padding=1),
            BatchNorm1d(afr_reduced_cnn_size),
            ReLU(),
            MaxPool1d(kernel_size=4, stride=1, padding=1)
        )

        # 初始化网络层的权重
        for m in self.modules():
            if isinstance(m, Conv1d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, BatchNorm1d):
                ones_(m.weight)
                zeros_(m.bias)

    def forward(self, x):
        """
        模型的前向传播。
        
        参数：
        x (torch.Tensor): 输入张量。
        
        返回：
        torch.Tensor: 通过卷积神经网络后的输出特征。
        """
        order_r = 12  # 滤波器阶数
        nyq = 0.5 * self.fs  # 奈奎斯特频率
        # 设计带通巴特沃斯滤波器
        c, d = signal.butter(order_r, [self.lowFr / nyq, self.higFr / nyq], btype='bandpass')

        # 将输入张量重塑为二维数组（batch_size, num_samples）
        x = x.view(-1, x.shape[-1])
        if x.device.type == 'cuda':
            # 将张量移到CPU并转换为NumPy数组
            x = x.cpu().numpy()

        assert isinstance(x, ndarray), "x应该是NumPy ndarray"

        # 对输入信号应用带通滤波器
        x = signal.filtfilt(c, d, x)

        all_Pxx = []  # 用于存储功率谱密度的列表
        for data in x:
            # 使用Welch方法计算功率谱密度
            _, Pxx = welch(data, fs=self.fs, nperseg=self.nperseg)
            # 将功率谱密度转换为PyTorch张量
            Pxx_tensor = from_numpy(Pxx).float().to(self.device)
            all_Pxx.append(Pxx_tensor)

        # 将所有功率谱密度张量堆叠成一个张量
        all_Pxx = stack(all_Pxx)
        all_Pxx = all_Pxx.unsqueeze(1)  # 添加一个通道维度
        x = self.features1(all_Pxx)  # 将张量传递通过卷积神经网络

        return x


############################

class getDEFeature(Module):
    """
    用于提取DE特征的类。

    该类继承自nn.Module，实现了从输入信号中提取差分熵(DE)特征的方法。
    
    参数:
    - fs: 采样频率，默认为1000Hz。
    - nperseg: 每个段的样本数，默认为100。
    - lowFr: 低通滤波器的下限频率，默认为80Hz。
    - higFr: 高通滤波器的上限频率，默认为450Hz。
    - device: 计算设备，默认使用cuda。
    """

    def __init__(self, fs=1000, nperseg=100, lowFr=80, higFr=450,
                 device=device("cuda")):

        super(getDEFeature, self).__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.lowFr = lowFr
        self.higFr = higFr
        self.device = device

    def forward(self, x):
        """
        前向传播方法，用于提取输入信号的DE特征。

        参数:
        - x: 输入的时域信号，形状为(batch_size, time_length)。

        返回:
        - de_features: 提取的DE特征，形状为(batch_size, 1)。
        """
        # 定义滤波器的阶数和奈奎斯特频率
        order_r = 12  # 滤波器阶数
        nyq = 0.5 * self.fs  # 奈奎斯特频率

        # 设计带通巴特沃斯滤波器
        c, d = signal.butter(order_r, [self.lowFr / nyq, self.higFr / nyq], btype='bandpass')

        # 调整输入信号的形状，准备进行滤波
        x = x.view(-1, x.shape[-1])

        # 将张量转换为numpy数组以进行滤波处理
        if x.device.type == 'cuda':
            x = x.cpu().numpy()

        # 确保x是NumPy数组
        assert isinstance(x, ndarray), "x应该是NumPy ndarray"

        # 应用带通滤波器
        x = signal.filtfilt(c, d, x, axis=-1)

        # 初始化存储差分熵的列表
        de_features = []

        # 遍历处理后的每个信号，计算其差分熵
        for data in x:
            # 使用Welch方法计算功率谱密度
            _, Pxx = welch(data, fs=self.fs, nperseg=self.nperseg)

            # 计算差分熵
            # 避免除零错误，将Pxx中的零值替换为极小值
            Pxx = where(Pxx == 0, finfo(float).eps, Pxx)
            log_Pxx = log(Pxx)
            de = -sum(Pxx * log_Pxx)

            # 将差分熵转换为PyTorch张量
            de_tensor = tensor(de, dtype=float32).to(self.device)
            de_features.append(de_tensor)

        # 将差分熵张量堆叠成一个张量
        de_features = stack(de_features)
        return de_features


############################
class getPsdValue(Module):
    def __init__(self, fs=1000, nperseg=100, lowFr=80, higFr=450,
                 device=device("cuda")):
        """
        初始化getPsdValue类。
        
        参数：
        fs (int): 采样频率。
        nperseg (int): 每个段的长度，用于Welch方法。
        lowFr (int): 带通滤波器的低频边界。
        higFr (int): 带通滤波器的高频边界。
        device (torch.device): 运行模型的设备（CPU或CUDA）。
        """
        super(getPsdValue, self).__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.lowFr = lowFr
        self.higFr = higFr
        self.device = device

    def forward(self, x):
        order_r = 12  # 滤波器阶数
        nyq = 0.5 * self.fs  # 奈奎斯特频率
        # 设计带通巴特沃斯滤波器
        c, d = signal.butter(order_r, [self.lowFr / nyq, self.higFr / nyq], btype='bandpass')

        # 将输入张量重塑为二维数组（batch_size, num_samples）
        x = x.view(-1, x.shape[-1])
        if x.device.type == 'cuda':
            # 将张量移到CPU并转换为NumPy数组
            x = x.cpu().numpy()

        assert isinstance(x, ndarray), "x应该是NumPy ndarray"

        # 对输入信号应用带通滤波器
        x = signal.filtfilt(c, d, x)

        all_avg_Pxx = []  # 用于存储平均功率谱密度的列表
        for data in x:
            # 使用Welch方法计算功率谱密度
            _, Pxx = welch(data, fs=self.fs, nperseg=self.nperseg)
            # 计算平均功率谱密度
            avg_Pxx = mean(Pxx)
            # 将平均功率谱密度转换为PyTorch张量
            avg_Pxx_tensor = tensor(avg_Pxx).float().to(self.device)
            all_avg_Pxx.append(avg_Pxx_tensor)

        # 将所有平均功率谱密度张量堆叠成一个张量
        all_avg_Pxx = stack(all_avg_Pxx)

        return all_avg_Pxx


#############################################


class getESDFeature(Module):
    """
    用于提取能量谱密度(ESD)特征的类。
    
    参数:
    - fs: 采样频率，默认为1000Hz。
    - nperseg: 每个段的样本数，默认为100。
    - lowFr: 低通滤波器的下限频率，默认为80Hz。
    - higFr: 高通滤波器的上限频率，默认为450Hz。
    - device: 计算设备，默认使用cuda。
    """

    def __init__(self, fs=1000, nperseg=100, lowFr=80, higFr=450,
                 device=device("cuda")):

        super(getESDFeature, self).__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.lowFr = lowFr
        self.higFr = higFr
        self.device = device

    def forward(self, x):
        """
        前向传播方法，用于提取输入信号的ESD特征。
        
        参数:
        - x: 输入的时域信号，形状为(batch_size, time_length)。
        
        返回:
        - esd_features: 提取的ESD特征，形状为(batch_size, 1)。
        """
        # 定义滤波器的阶数和奈奎斯特频率
        order_r = 12  # 滤波器阶数
        nyq = 0.5 * self.fs  # 奈奎斯特频率

        # 设计带通巴特沃斯滤波器
        c, d = signal.butter(order_r, [self.lowFr / nyq, self.higFr / nyq], btype='bandpass')

        # 调整输入信号的形状，准备进行滤波
        x = x.view(-1, x.shape[-1])

        # 将张量转换为numpy数组以进行滤波处理
        if x.device.type == 'cuda':
            x = x.cpu().numpy()

        # 确保x是NumPy数组
        assert isinstance(x, ndarray), "x应该是NumPy ndarray"

        # 应用带通滤波器
        x = signal.filtfilt(c, d, x, axis=-1)

        # 初始化存储能量谱密度的列表
        esd_features = []

        # 遍历处理后的每个信号，计算其能量谱密度
        for data in x:
            # 使用Welch方法计算能量谱密度
            f, Sxx = welch(data, fs=self.fs, nperseg=self.nperseg, scaling='spectrum')

            # 计算总能量
            total_energy = trapz(Sxx, f)

            # 将总能量转换为PyTorch张量
            energy_tensor = tensor(total_energy, dtype=float32).to(self.device)
            esd_features.append(energy_tensor)

        # 将能量谱密度张量堆叠成一个张量
        esd_features = stack(esd_features)
        return esd_features

# if __name__=="__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_data = torch.randn(32, 1, 100).to(device)
#     getpsdvalue = getPsdFeature().to(device)
#     output = getpsdvalue(input_data)
#     print(output.size())
