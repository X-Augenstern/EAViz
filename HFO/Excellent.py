from torch import device, cuda, roll, cat, zeros, float32, tensor
from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, GELU, MaxPool1d, Dropout
from torch.nn.init import xavier_uniform_, zeros_, ones_
from HFO.modules import getESDFeature, getDEFeature, getPsdFeature, getPsdValue
from HFO.timFeature import downSample, SEBasicBlock, makeLayer
from HFO.Attention0 import TransformerAEClassifier
from scipy.signal import butter, filtfilt

device = device("cuda" if cuda.is_available() else "cpu")


##########################################################################################
class NLEO(Module):
    def __init__(self):
        super(NLEO, self).__init__()

    def forward(self, x):
        x_shifted_left = roll(x, shifts=-1, dims=-1)  # 向左平移一位
        x_shifted_right = roll(x, shifts=1, dims=-1)  # 向右平移一位
        nleo = x ** 2 - x_shifted_left * x_shifted_right  # 计算SNEO
        return nleo


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut=1.0, highcut=70.0, fs=1000, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y


def pad_signal(signal, pad_length=1):
    # 使用边界值进行填充
    left_pad = signal[..., :pad_length]
    right_pad = signal[..., -pad_length:]
    padded_signal = cat([left_pad, signal, right_pad], dim=-1)
    return padded_signal


class NLEO_Get(Module):
    def __init__(self):
        super(NLEO_Get, self).__init__()
        self.NLEO = NLEO()

    def forward(self, x):
        x_filtered = bandpass_filter(x.cpu().numpy()).copy()
        x_filtered = tensor(x_filtered, dtype=float32).to(device)
        x_padded = pad_signal(x_filtered, pad_length=1)
        x_sneo = self.NLEO(x_padded)
        x_sneo = x_sneo[:, :, 1:-1]
        return x_sneo


##########################################################################################
class MRCNN(Module):
    def __init__(self):
        super(MRCNN, self).__init__()
        drate = 0.5

        self.features1_1 = Sequential(
            Conv1d(1, 64, kernel_size=2, stride=1, bias=False, padding=1),
            BatchNorm1d(64)
        )

        self.features1_2 = Sequential(
            GELU(),
            MaxPool1d(kernel_size=4, stride=2, padding=1),
            Dropout(drate),

            Conv1d(64, 128, kernel_size=4, stride=1, bias=False, padding=1),
            BatchNorm1d(128),
            GELU(),

            Conv1d(128, 128, kernel_size=4, stride=1, bias=False, padding=1),
            BatchNorm1d(128),
            GELU(),

            MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.features2_1 = Sequential(
            Conv1d(1, 64, kernel_size=14, stride=1, bias=False, padding=1),
            BatchNorm1d(64)
        )

        self.features2_2 = Sequential(
            GELU(),
            MaxPool1d(kernel_size=2, stride=2, padding=1),
            Dropout(drate),

            Conv1d(64, 128, kernel_size=2, stride=1, bias=False, padding=1),
            BatchNorm1d(128),
            GELU(),

            Conv1d(128, 128, kernel_size=2, stride=1, bias=False, padding=1),
            BatchNorm1d(128),
            GELU(),

            MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        self.dropout = Dropout(drate)

        self.features3 = downSample(SEBasicBlock, 128, 32)

        for m in self.modules():
            if isinstance(m, Conv1d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, BatchNorm1d):
                ones_(m.weight)
                zeros_(m.bias)

    def forward(self, x):
        x1_1 = self.features1_1(x)
        x1_2 = self.features1_2(x1_1)

        x2_1 = self.features2_1(x)
        x2_2 = self.features2_2(x2_1)

        x1_mean = x1_2.mean(dim=0, keepdim=True)
        x1_std = x1_2.std(dim=0, keepdim=True)
        x1_normalized = (x1_2 - x1_mean) / (x1_std + 1e-5)

        x2_mean = x2_2.mean(dim=0, keepdim=True)
        x2_std = x2_2.std(dim=0, keepdim=True)
        x2_normalized = (x2_2 - x2_mean) / (x2_std + 1e-5)

        x_concat = cat((x1_normalized, x2_normalized), dim=2)
        x_concat = self.dropout(x_concat)
        out = self.features3(x_concat)

        return out


##############################################
# PSD加权NLEO
class PN(Module):
    def __init__(self):
        super(PN, self).__init__()
        self.inplanes = 32
        self.planes = 32
        afr_reduced_cnn_size = 32
        self.SNEO = NLEO_Get()
        self.mfc = MRCNN()
        self.getpsdfeature = getPsdFeature()
        self.mL = makeLayer(self.inplanes, self.planes)
        self.At = TransformerAEClassifier()

    def forward(self, x):
        x = self.SNEO(x)
        x_feature = self.mfc(x)
        # print(x_feature.size())
        inPsd = self.getpsdfeature(x)
        out_tim, out_psd = self.mL(x_feature, inPsd)
        out = cat([out_psd, out_tim, out_psd], dim=-1)
        out = self.At(out)

        return out


#############################################
# 使用51个频段PSD
#   main函数
class Celestial(Module):
    def __init__(self):
        super(Celestial, self).__init__()
        self.inplanes = 32
        self.planes = 32
        self.mfc = MRCNN()
        self.getpsdfeature = getPsdFeature()
        self.mL = makeLayer(self.inplanes, self.planes)
        self.At = TransformerAEClassifier()

    def forward(self, x):
        x_feature = self.mfc(x)
        # print(x_feature.size())
        inPsd = self.getpsdfeature(x)
        out_tim, out_psd = self.mL(x_feature, inPsd)
        out = cat([out_psd, out_tim, out_psd], dim=-1)
        print(out.size())
        print("forward")
        out = self.At(out)

        return out


#########################################
# 使用平均PSD
class Ce(Module):
    def __init__(self):
        super(Ce, self).__init__()
        self.inplanes = 32
        self.planes = 32
        self.mfc = MRCNN()
        self.getpsdfeature = getPsdValue()
        self.At = TransformerAEClassifier()

    def forward(self, x):
        x_feature = self.mfc(x)
        # print(x_feature.shape)
        inPsd = self.getpsdfeature(x)
        # print(inPsd.shape)
        # 将 inPsd 的形状扩展为 [32, 1, 1]
        inPsd_expanded = inPsd.view(-1, 1, 1)
        # print(inPsd_expanded.shape)  # 输出: torch.Size([32, 1, 1])

        # 利用广播机制使 inPsd 与 x_feature 相乘
        result = x_feature * inPsd_expanded
        # print(result.shape)  # 输出: torch.Size([32, 32, 96])
        out = self.At(result)

        return out


#########################################

# 使用DE
class DDDE(Module):
    def __init__(self):
        super(DDDE, self).__init__()
        self.inplanes = 32
        self.planes = 32
        self.mfc = MRCNN()
        self.getpsdfeature = getDEFeature()
        self.At = TransformerAEClassifier()

    def forward(self, x):
        x_feature = self.mfc(x)
        # print(x_feature.shape)
        inPsd = self.getpsdfeature(x)
        # print(inPsd.shape)
        # 将 inPsd 的形状扩展为 [32, 1, 1]
        inPsd_expanded = inPsd.view(-1, 1, 1)
        # print(inPsd_expanded.shape)  # 输出: torch.Size([32, 1, 1])

        # 利用广播机制使 inPsd 与 x_feature 相乘
        result = x_feature * inPsd_expanded
        # print(result.shape)  # 输出: torch.Size([32, 32, 96])

        # 在最后一个维度上添加两个全零的特征
        result = cat([result, zeros(result.size(0), result.size(1), 2, device=result.device)], dim=2)
        # print(result.size())  # 输出: torch.Size([32, 32, 98])

        out = self.At(result)
        # print("OK***************************")

        return out


#####################################################################

# 使用ESD
class ESDD(Module):
    def __init__(self):
        super(ESDD, self).__init__()
        self.inplanes = 32
        self.planes = 32
        self.mfc = MRCNN()
        self.getpsdfeature = getESDFeature()
        self.At = TransformerAEClassifier()

    def forward(self, x):
        x_feature = self.mfc(x)
        # print(x_feature.shape)
        inPsd = self.getpsdfeature(x)
        # print(inPsd.shape)
        # 将 inPsd 的形状扩展为 [32, 1, 1]
        inPsd_expanded = inPsd.view(-1, 1, 1)
        # print(inPsd_expanded.shape)  # 输出: torch.Size([32, 1, 1])

        # 利用广播机制使 inPsd 与 x_feature 相乘
        result = x_feature * inPsd_expanded
        # print(result.shape)  # 输出: torch.Size([32, 32, 96])

        # 在最后一个维度上添加两个全零的特征
        result = cat([result, zeros(result.size(0), result.size(1), 2, device=result.device)], dim=2)
        # print(result.size())  # 输出: torch.Size([32, 32, 98])

        out = self.At(result)
        # print("OK***************************")

        return out

##########################################################################################
# if __name__ == "__main__":
#     import os
#
#     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_data = torch.randn(32, 1, 100).to(device)
#     model = ESDD().to(device)
#
#     output = model(input_data)
#     print(output.size())
