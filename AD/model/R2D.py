from torch.nn import Module, BatchNorm1d, Conv1d, MaxPool1d, AdaptiveAvgPool1d, Linear, ReLU, Sequential
from torch.nn.functional import relu

# from torchsummary import summary

Conv = Conv1d
BN = BatchNorm1d


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BN(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BN(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(Module):
    def __init__(self, block, num_blocks, loss=None, in_channel=10,
                 channel_list=[64, 128, 256, 512], out_index=[0, 1, 2, 3]):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.loss = loss
        self.out_index = out_index
        self.conv1 = Conv(in_channel, self.in_planes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        self.bn1 = BN(self.in_planes)
        self.relu = ReLU(inplace=True)
        # self.sigmoid = Sigmoid()
        self.maxpool = MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channel_list[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channel_list[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channel_list[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, channel_list[3], num_blocks[3], stride=2)
        self.avgpool = AdaptiveAvgPool1d(1)
        self.fc1 = Linear(512, 128)
        self.fc2 = Linear(128, 6)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.squeeze(-1)
        return self.fc2(self.fc1(out))


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# if __name__ == '__main__':
#     input = rand(8, 10, 1000)
#     model = resnet34().cuda()
#     summary(model, (10, 1000))
#     print(model(input).shape)
