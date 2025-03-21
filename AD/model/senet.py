'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
from torch import randn
from torch.nn import Module, BatchNorm1d, Conv1d, MaxPool1d, AdaptiveAvgPool1d, Linear, ReLU, Sequential
from torch.nn.functional import relu, sigmoid, avg_pool2d

Conv = Conv1d
BN = BatchNorm1d


class BasicBlock(Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BN(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = Sequential(
                Conv(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                BN(planes)
            )

        # SE layers
        self.fc1 = Conv(planes, planes // 16, kernel_size=1)  # Use Conv instead of nn.Linear
        self.fc2 = Conv(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = avg_pool2d(out, out.size(2))
        w = relu(self.fc1(w))
        w = sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = relu(out)
        return out


class PreActBlock(Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = BN(in_planes)
        self.conv1 = Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = Sequential(
                Conv(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = Conv(planes, planes // 16, kernel_size=1)
        self.fc2 = Conv(planes // 16, planes, kernel_size=1)

        self.avgpool = AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(relu(self.bn2(out)))

        # Squeeze
        w = self.avgpool(out)
        w = relu(self.fc1(w))
        w = sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(Module):
    def __init__(self, block, num_blocks, loss=None, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64
        self.loss = loss

        self.conv1 = Conv(10, self.in_planes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        self.bn1 = BN(self.in_planes)
        self.relu = ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.maxpool = MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear(512, 6)

        self.avgpool = AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.squeeze(-1)
        out = self.linear(out)
        return out


def SENet18(**kwargs):
    return SENet(PreActBlock, [2, 2, 2, 2], **kwargs)


def test():
    net = SENet18()
    y = net(randn(1, 3, 32, 32))
    print(y.size())


# if __name__ == "__main__":
#     from torchsummary import summary
#
#     model = SENet18().cuda()
#     summary(model, (10, 1000))

# test()
