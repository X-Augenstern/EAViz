from torch.nn.modules.utils import _triple
# from torchsummary import summary
# from thop import profile
from torch.nn import Sigmoid, Linear, ReLU, Sequential, Module, AdaptiveAvgPool2d, Conv3d, BatchNorm3d, ModuleList, \
    MaxPool3d, AdaptiveAvgPool3d, init


# from torch.autograd import Variable
# from torchsummary import summary


class SELayer(Module):
    def __init__(self, out_channels, reduction=16):
        super(SELayer, self).__init__()
        self.cse_avg_pool = AdaptiveAvgPool2d(1)
        self.cse_fc = Sequential(
            Linear(out_channels, out_channels * reduction, bias=False),
            ReLU(inplace=True),
            Linear(out_channels * reduction, out_channels, bias=False),
            Sigmoid()
        )

    def forward(self, x):
        b, c, d, e = x.size()
        x = x.permute((0, 3, 2, 1))
        x = self.cse_avg_pool(x)
        x = x.view(b, -1)
        cse_y = self.cse_fc(x).view(b, 1, e, 1, 1)

        return cse_y


class SpatioTemporalConv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        self.temporal_spatial_conv = Conv3d(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, bias=bias)
        self.bn = BatchNorm3d(out_channels)
        self.relu = ReLU()

    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, output, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = BatchNorm3d(out_channels)
        self.relu1 = ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = BatchNorm3d(out_channels)
        self.outrelu = ReLU()

        self.se = SELayer(output)

    def forward(self, x, y):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        res = res * self.se(y)

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, output, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, output, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size, output)]

    def forward(self, x, y):
        x = self.block1(x, y)
        for block in self.blocks:
            x = block(x, y)

        return x


class R3DNet(Module):
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.maxpool = MaxPool3d(kernel_size=2, stride=2, padding=1)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        return x.view(-1, 512)


class R3DClassifier(Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R3DClassifier, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)

        self.age_fc_layers = Sequential(
            Linear(512, 256),
            ReLU(),
            Linear(256, 2),
            Sigmoid()
        )

        # 预测gender（分类）
        self.gender_fc_layers = Sequential(
            Linear(512, 256),
            ReLU(),
            Linear(256, 2)
        )

        # self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):  # 原始训练网络
        x = self.res3d(x)
        # logits = self.linear(x)
        # outputs = self.linear(x)
        out_age = self.age_fc_layers(x)
        out_gender = self.gender_fc_layers(x)
        return out_age, out_gender
        # return outputs

    # def forward(self, x):
    #     outputs = []
    #
    #     for name, module in self.res3d.named_children():  # 提取res3d卷积特征
    #         x = module(x)
    #         # if name in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
    #         if name in ["conv2"]:
    #             outputs.append(x)
    #     return outputs

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, Conv3d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

# if __name__ == "__main__":
#     import torch
#     inputs = torch.rand(1, 3, 21, 32, 32)
#     net = R3DClassifier(2, (2, 2, 2, 2), pretrained=True)#(2,2,2,2)为存储layer_sizes列表的个数，layer_size[1]对应conv1残差，2为个数
#
#     outputs = net.forward(inputs)
#     # print(outputs.size())
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net = net.to(device)
#     summary(net, (3, 21, 32, 32))
#
#     # flops, params = profile(net, (inputs,))
#     # print("%.3fG" % (flops / 1000 ** 3), "%.3fM" % (params / 1000 ** 2))
