from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, AdaptiveAvgPool1d, Linear, ReLU, Sigmoid
from HFO.modules import psdWeight


def downSample(block, inplanes, planes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = Sequential(
            Conv1d(inplanes, planes * block.expansion,
                   kernel_size=1, stride=stride, bias=False),
            BatchNorm1d(planes * block.expansion),
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))

    return Sequential(*layers)


# SE层中包含乘法的部分，即一个自适应池化和两个全连接
class SELayer(Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = AdaptiveAvgPool1d(1)

        self.fc = Sequential(
            Linear(channel, channel // reduction, bias=False),
            ReLU(inplace=True),
            Linear(channel // reduction, channel, bias=False),
            Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # 获取输入张量 x 的形状信息，其中 b 表示 batch size，c 表示通道数
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)  # 将y的形状进行扩展，使得和x的形状相同，再相乘 
    # SE模块完成


# SE层中包含卷积和残差的部分
class SEBasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = Conv1d(inplanes, planes, stride)
        self.bn1 = BatchNorm1d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv1d(planes, planes, 1)
        self.bn2 = BatchNorm1d(planes)
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


class SEAdvancedBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEAdvancedBlock, self).__init__()
        self.seBB = SEBasicBlock(inplanes, planes)
        self.pW = psdWeight(planes, reduction)
        self.conv = Conv1d(planes, planes, 1)
        self.bn = BatchNorm1d(planes)

    def forward(self, inTim, inPsd):
        inTimFeature = self.seBB(inTim)
        # inTimFeature = self.conv(inTimFeature)
        # inTimFeature = self.bn(inTimFeature)
        out, y = self.pW(inPsd, inTimFeature)
        out = out + inTim
        return out, y


##########################################################################################
class makeLayer(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(makeLayer, self).__init__()
        self.SEAB = SEAdvancedBlock(inplanes, planes)

    def forward(self, inTim, inPsd):
        out, y = self.SEAB(inTim, inPsd)
        out, y = self.SEAB(out, y)

        return out, y

##########################################################################################
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_data = torch.randn(32, 32, 96).to(device)
#     Psd = torch.randn(32, 32, 48).to(device)
#     model = makeLayer(32, 32).to(device)
#     # getpsdfeature = modules.getPsdFeature().to(device)
#
#     output, y = model(input_data, Psd)
#     print(output.size())
#     print(y.size())
