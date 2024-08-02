'''GoogLeNet with PyTorch.'''
from torch import cat, randn
from torch.nn import Module, BatchNorm1d, Conv1d, MaxPool1d, AvgPool1d, Linear, ReLU, Sequential

Conv = Conv1d
BN = BatchNorm1d


class Inception(Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = Sequential(
            Conv(in_planes, n1x1, kernel_size=1),
            BN(n1x1),
            ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = Sequential(
            Conv(in_planes, n3x3red, kernel_size=1),
            BN(n3x3red),
            ReLU(True),
            Conv(n3x3red, n3x3, kernel_size=3, padding=1),
            BN(n3x3),
            ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = Sequential(
            Conv(in_planes, n5x5red, kernel_size=1),
            BN(n5x5red),
            ReLU(True),
            Conv(n5x5red, n5x5, kernel_size=3, padding=1),
            BN(n5x5),
            ReLU(True),
            Conv(n5x5, n5x5, kernel_size=3, padding=1),
            BN(n5x5),
            ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = Sequential(
            MaxPool1d(3, stride=1, padding=1),
            Conv(in_planes, pool_planes, kernel_size=1),
            BN(pool_planes),
            ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return cat([y1, y2, y3, y4], 1)


class GoogLeNet(Module):
    def __init__(self, loss=None):
        super(GoogLeNet, self).__init__()
        self.loss = loss
        self.pre_layers = Sequential(
            Conv(64, 192, kernel_size=3, padding=1),
            BN(192),
            ReLU(True),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = MaxPool1d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = AvgPool1d(8)
        self.linear = Linear(1024 * 7, )

        self.in_planes = 64
        self.conv1 = Conv(10, self.in_planes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        self.bn1 = BN(self.in_planes)
        self.relu = ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.maxpool1 = MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        out = self.pre_layers(out)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = GoogLeNet()
    x = randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())

# if __name__ == "__main__":
#     from torchsummary import summary
#     model = GoogLeNet().cuda()
#     summary(model, (10, 1000))

# test()
