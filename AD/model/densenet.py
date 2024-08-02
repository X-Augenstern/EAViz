'''DenseNet in PyTorch.'''
from math import floor
from torch import cat, randn
from torch.nn import Module, BatchNorm1d, Conv1d, MaxPool1d, AdaptiveAvgPool1d, Linear, ReLU, Sequential
from torch.nn.functional import relu, avg_pool1d

Conv = Conv1d
BN = BatchNorm1d


class Bottleneck(Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = BatchNorm1d(in_planes)
        self.conv1 = Conv1d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = BatchNorm1d(4 * growth_rate)
        self.conv2 = Conv1d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(relu(self.bn1(x)))
        out = self.conv2(relu(self.bn2(out)))
        out = cat([out, x], 1)
        return out


class Transition(Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = BatchNorm1d(in_planes)
        self.conv = Conv1d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(relu(self.bn(x)))
        out = avg_pool1d(out, 2)
        return out


class DenseNet(Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, loss=None, num_classes=6):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.loss = loss

        num_planes = 2 * growth_rate
        self.conv1 = Conv1d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.in_planes = 64
        self.bn = BatchNorm1d(num_planes)
        self.linear = Linear(num_planes, num_classes)

        self.conv1 = Conv(10, self.in_planes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        self.bn1 = BN(self.in_planes)
        self.relu = ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.maxpool = MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = AdaptiveAvgPool1d(1)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = relu(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def DenseNet121(**kwargs):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, **kwargs)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


def test():
    net = densenet_cifar()
    x = randn(1, 3, 32, 32)
    y = net(x)
    print(y)


if __name__ == "__main__":
    from torchsummary import summary

    model = DenseNet121().cuda()
    summary(model, (10, 1000))

# test()
