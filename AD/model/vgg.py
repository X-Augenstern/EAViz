'''VGG11/13/16/19 in Pytorch.'''
from torch import randn
from torch.nn import Module, BatchNorm1d, Conv1d, MaxPool1d, AdaptiveAvgPool1d, Linear, ReLU, Sequential, AvgPool1d

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

Conv = Conv1d
BN = BatchNorm1d


class VGG(Module):
    def __init__(self, vgg_name, loss=None):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = Linear(512, 6)

        self.loss = loss;

        self.in_planes = 64

        self.conv1 = Conv(10, self.in_planes, kernel_size=7, stride=2, padding=3,
                          bias=False)
        self.bn1 = BN(self.in_planes)
        self.relu = ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.maxpool = MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.features(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 64
        for x in cfg:
            if x == 'M':
                layers += [MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [Conv1d(in_channels, x, kernel_size=3, padding=1),
                           BatchNorm1d(x),
                           ReLU(inplace=True)]
                in_channels = x
        layers += [AvgPool1d(kernel_size=1, stride=1)]
        return Sequential(*layers)


def VGG16(**kwargs):
    return VGG('VGG16', **kwargs)


def test():
    net = VGG('VGG11')
    x = randn(1, 10, 1000)
    y = net(x)
    print(y.size())


# if __name__ == "__main__":
#     from torchsummary import summary
#
#     model = VGG('VGG16', loss=None).cuda()
#     summary(model, (10, 1000))
#     test()

# test()
