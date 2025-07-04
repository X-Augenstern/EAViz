from torch.nn import Module, AdaptiveAvgPool3d, Sequential, Linear, ReLU, Sigmoid, Conv3d, BatchNorm3d, ModuleList, \
    MaxPool3d, init
from torch.nn.modules.utils import _triple


class SELayer(Module):
    """
    1.squeeze: 全局池化 (batch,channel,height,width) -> (batch,channel,1,1) ==> (batch,channel)
    2.excitation: 全连接or卷积核为1的卷积(batch,channel) -> (batch,channel//reduction) -> (batch,channel) ==> (batch,channel,1,1) 输出y
    3.scale: 完成对通道维度上原始特征的标定 y = x*y 输出维度和输入维度相同
    """

    def __init__(self, out_channels, reduction=16):
        super(SELayer, self).__init__()
        # squeeze 自适应全局平均池化,即每个通道进行平均池化，使输出特征图长宽为1
        self.cse_avg_pool = AdaptiveAvgPool3d(1)
        # 全连接的excitation
        self.cse_fc = Sequential(
            Linear(out_channels, out_channels // reduction, bias=False),
            ReLU(inplace=True),
            Linear(out_channels // reduction, out_channels, bias=False),
            Sigmoid()
        )

    def forward(self, x):
        # (batch,channel,height,width) (2,512,8,8,8)
        b, c, z, w, h = x.size()
        # squeeze (2,512,8,8,8) -> (2,512,1,1,1) -> (2,512)
        cse_y = self.cse_avg_pool(x).view(b, c)
        # excitation (2,512) -> (2,512//reduction) -> (2,512) -> (2,512,1,1,1)
        cse_y = self.cse_fc(cse_y).view(b, c, 1, 1, 1)
        # scale (2,512,8,8,8) * (2,512,1,1,1) -> (2,512,8,8,8)
        return x * cse_y.expand_as(x)


class SpatioTemporalConv(Module):
    """
    Conv3d - BatchNorm3d - ReLU
    """

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
    """
    res = Conv3d - BatchNorm3d - ReLU | - BatchNorm3d - ReLU | - Conv3d - BatchNorm3d - ReLU | - BatchNorm3d - SE
    x+res - ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
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

        self.se = SELayer(out_channels)

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        res = self.se(res)

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(Module):
    """
    SpatioTemporalResBlock * layer_size
    """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class R3DNet(Module):
    """
    Conv3d - BatchNorm3d - ReLU | - MaxPool3d
    - SpatioTemporalResBlock * layer_size[0]
    - SpatioTemporalResBlock * layer_size[1]
    - SpatioTemporalResBlock * layer_size[2]
    - SpatioTemporalResBlock * layer_size[3] - AdaptiveAvgPool3d
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.maxpool = MaxPool3d(kernel_size=2, stride=2, padding=1)
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)  # Attention Block
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = AdaptiveAvgPool3d(1)

    def forward(self, x):
        # new addition
        feature_map = []

        x = self.conv1(x)
        feature_map.append(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        feature_map.append(x)
        x = self.conv3(x)
        feature_map.append(x)
        x = self.conv4(x)
        feature_map.append(x)
        x = self.conv5(x)
        feature_map.append(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        return x.view(-1, 512), feature_map


class R3DClassifier(Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

    R3DNet - Linear

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R3DClassifier, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x, feature_map = self.res3d(x)
        # logits = self.linear(x)
        outputs = self.linear(x)
        # out_age = self.age_fc_layers(x)
        # out_gender = self.gender_fc_layers(x)
        return outputs, feature_map
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
#
#     inputs = torch.rand(16, 1, 21, 32, 32)
#     model = R3DClassifier(7, (2, 2, 2, 2), pretrained=True)  # (2,2,2,2)为存储layer_sizes列表的个数，layer_size[1]对应conv1残差，2为个数
#     train_params = model.parameters()
#     # optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
#     optimizer = optim.Adam(train_params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#     torch.save({
#         'epoch': 1,
#         'state_dict': model.state_dict(),
#         'opt_dict': optimizer.state_dict(),
#     }, os.path.join('test' + '.pth.tar'))
#     print("Save model at {}\n".format(
#         os.path.join('test' + '.pth.tar')))
#     outputs = model(inputs)
#     print(outputs.size())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = net.to(device)
# summary(net, (1, 45, 45, 6))

# flops, params = profile(net, (inputs,))
# print("%.3fG" % (flops / 1000 ** 3), "%.3fM" % (params / 1000 ** 2))
