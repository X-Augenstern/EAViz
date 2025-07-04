from einops import rearrange  # rearrange是einops中的一个函数调用方法。einops的强项是把张量的维度操作具象化，可以一次将多个维度转换，类似于permute，比transpose一次只能调换两个维度效率要高
from torch import nn, bmm
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

Conv = nn.Conv1d
BN = nn.BatchNorm1d
MaxPool = nn.MaxPool1d
AvgPool = nn.AdaptiveAvgPool1d


class SELayer(nn.Module):
    def __init__(self, channel, increase=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # b*c*32*5000 -> b*c*1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * increase, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * increase, channel, bias=False),  # 两层的顺序调整 先降就太少了
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # b*c*1*1 -> b*C  12*8
        att = self.fc(y)  # b*c（特征） -> b*c（权重）
        return att.unsqueeze(1)  # 插入维度 b*c -> b*1*c  12 1 8


class SignalAttention(nn.Module):
    def __init__(self, planes, increase=8):
        super(SignalAttention, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1)
        self.filters = filters = [32, 64, 128, 256, 512]
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, increase)

    def forward(self, x, feat, batch):
        out = self.conv1(x)  # x:b*c*512*312 -> b, c, 256, 156  输入是hd5 96 512 312 展后 12 8 512 312
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # b, c, 256, 156-> b, c, 256, 156
        out = self.bn2(out)
        att = self.se(out)  # b, c, 256, 156-> b, 1, c  batch中的每一个通道得到一个1*8的权重向量
        for i in range(len(feat)):
            if feat[i] is not None:
                feat[i] = bmm(att, rearrange(feat[i], '(B S) C T -> B S (C T)', B=batch)).view(batch, self.filters[i], -1)
        return att, feat


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, last=False):
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=0 if last else dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, last=False):
    return Conv(in_planes, out_planes, kernel_size=3 if last else 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, last=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride, last=last)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, last=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, initChannel=19, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, SA=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.dilation = 1
        self.SA = SA
        if SA:
            self.seg_attention = SignalAttention(19)
            self.stage_attention = SignalAttention(19)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv(initChannel, self.inplanes, kernel_size=5, stride=3, padding=1,
                          bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], last=True)
        for m in self.modules():
            if isinstance(m, Conv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BN, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, last=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, last=last),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, last=last)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch, channel, time = x.shape
        if self.SA:
            x = x.view(batch * channel, 1, time)
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)     # [12, 32, 5000]

        x2 = self.layer1(x1)  # [12, 64, 5000]
        x3 = self.layer2(x2)  # [12, 128,1250]
        x4 = self.layer3(x3)  # [12, 256, 625]
        x5 = self.layer4(x4)  # [12, 512, 312]

        feat1 = [x1, x2, x3, x4, x5]
        feat2 = [None, None, None, None, x5 + 0]
        if self.SA:   # returen att, feat 返回 特征权重  以及 加权后的特征
            seg_att, feat1 = self.seg_attention(feat1[-1].view(batch, -1, *feat1[-1].shape[1:]), feat1, batch)
            stage_att, feat2 = self.stage_attention(feat2[-1].view(batch, -1, *feat2[-1].shape[1:]), feat2, batch)
            return {'seg_att': seg_att, 'seg_feat': feat1, 'stage_att': stage_att, 'stage_feat': feat2[-1]}
        else:
            return {'seg_att': None, 'seg_feat': feat1, 'stage_att': None, 'stage_feat': feat2[-1]}


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], **kwargs)


# if __name__ == '__main__':
#     import torch
#
#     model = resnet34().float().cuda()
#     input = torch.rand(12, 19, 15000).float().cuda()
#     out = model(input)
    # print(model)
    # print(out.shape)
    # print(type(out))
