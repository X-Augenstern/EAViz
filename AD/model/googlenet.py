'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn

Conv = nn.Conv1d
BN = nn.BatchNorm1d

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            Conv(in_planes, n1x1, kernel_size=1),
            BN(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            Conv(in_planes, n3x3red, kernel_size=1),
            BN(n3x3red),
            nn.ReLU(True),
            Conv(n3x3red, n3x3, kernel_size=3, padding=1),
            BN(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            Conv(in_planes, n5x5red, kernel_size=1),
            BN(n5x5red),
            nn.ReLU(True),
            Conv(n5x5red, n5x5, kernel_size=3, padding=1),
            BN(n5x5),
            nn.ReLU(True),
            Conv(n5x5, n5x5, kernel_size=3, padding=1),
            BN(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            Conv(in_planes, pool_planes, kernel_size=1),
            BN(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self,loss = None):
        super(GoogLeNet, self).__init__()
        self.loss = loss
        self.pre_layers = nn.Sequential(
            Conv(64, 192, kernel_size=3, padding=1),
            BN(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool1d(8)
        self.linear = nn.Linear(1024*7,    )

        self.in_planes = 64
        self.conv1 = Conv(10, self.in_planes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        self.bn1 = BN(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

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
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    from torchsummary import summary
    model = GoogLeNet().cuda()
    summary(model, (10, 1000))

# test()
