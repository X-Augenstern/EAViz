from math import sqrt
from functools import reduce
from operator import mul
from numpy import finfo
from torch import mean, log, cat, Tensor, exp, randn_like, flatten, sigmoid, unsqueeze, load, transpose, clamp, gather, \
    squeeze
from torch.nn import MSELoss, L1Loss, Module, Conv1d, BatchNorm1d, LeakyReLU, ConvTranspose1d, Sequential, \
    AdaptiveMaxPool1d, Linear, AdaptiveAvgPool1d, CrossEntropyLoss, init, GroupNorm, Parameter, Dropout, Sigmoid
from torch.nn.functional import relu, linear, softmax, normalize


class ConvBlock(Module):
    def __init__(self, in_channel, mid_channel, out_channel, last_relu=True):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv1d(in_channel, mid_channel, 3, 1, 1)
        self.bn1 = BatchNorm1d(mid_channel)
        self.conv2 = Conv1d(mid_channel, out_channel, 3, 1, 1)
        self.bn2 = BatchNorm1d(out_channel)
        self.relu = LeakyReLU(inplace=True)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x) if self.last_relu else x


class DownSample(Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.conv = Conv1d(channel, channel, 3, 2, 1)
        self.bn = BatchNorm1d(channel)
        self.relu = LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class UpSample(Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.conv = ConvTranspose1d(channel, channel, 3, 2, 1, output_padding=1)
        self.bn = BatchNorm1d(channel)
        self.relu = LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class Encoder(Module):
    def __init__(self, skip=False):
        super(Encoder, self).__init__()
        self.block1 = ConvBlock(10, 32, 32)
        self.downblock1 = DownSample(32)
        self.block2 = ConvBlock(32, 64, 64)
        self.downblock2 = DownSample(64)
        self.block3 = ConvBlock(64, 128, 128)
        self.downblock3 = DownSample(128)
        self.block4 = ConvBlock(128, 256, 256)
        self.downblock4 = DownSample(256)
        self.block5 = ConvBlock(256, 512, 512)
        self.skip = skip

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.downblock1(x1)
        x2 = self.block2(x2)
        x3 = self.downblock2(x2)
        x3 = self.block3(x3)
        x4 = self.downblock3(x3)
        x4 = self.block4(x4)
        x5 = self.downblock4(x4)
        x5 = self.block5(x5)
        if self.skip:
            return x5, x4, x3, x2, x1
        else:
            return x5


class Decoder(Module):
    def __init__(self, with_last_relu=False):
        super(Decoder, self).__init__()
        self.block1 = ConvBlock(512, 256, 256)
        self.upblock1 = UpSample(256)
        self.block2 = ConvBlock(256, 128, 128)
        self.upblock2 = UpSample(128)
        self.block3 = ConvBlock(128, 64, 64)
        self.upblock3 = UpSample(64)
        self.block4 = ConvBlock(64, 32, 32)
        self.upblock4 = UpSample(32)
        self.block5 = ConvBlock(32, 10, 10, with_last_relu)
        # self.skipblock = SkipBlock(512, 256)

    def forward(self, x):
        x = self.block1(x)
        x = self.upblock1(x)
        x = self.block2(x)
        x = self.upblock2(x)
        x = self.block3(x)
        x = self.upblock3(x)
        x = self.block4(x)
        x = self.upblock4(x)
        x = self.block5(x)
        return x


# ---------------------------------------AutoEncoder--------------------------------------
class AutoEncoder(Module):
    def __init__(self, with_last_relu=False):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(with_last_relu)
        self.Loss = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, pred, true):
        reconloss = self.Loss(pred, true)
        return dict(Recon_Loss=reconloss, Total_Loss=reconloss)
        # CorrLoss = self.corr_loss(pred, true)
        # totalloss = CorrLoss + reconloss
        # return dict(Recon_Loss=reconloss, Corr_Loss=CorrLoss, Total_Loss=totalloss)

    def slope_loss(self, pred, true):
        pred_shifted_right = pred[:, :, 1:]
        pred_shifted_left = pred[:, :, :-1]
        pred_slope = abs(pred_shifted_left - pred_shifted_right)

        true_shift_right = true[:, :, 1:]
        true_shift_left = true[:, :, :-1]
        true_slope = abs(true_shift_left - true_shift_right)
        SlopeLoss = self.Loss(pred_slope, true_slope)
        return SlopeLoss


# ---------------------------------------AutoEncoder--------------------------------------

# ---------------------------------------Resnet_Encoder-----------------------------------
class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm1d(planes)
        self.conv2 = Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm1d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm1d(planes)
        self.conv2 = Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm1d(planes)
        self.conv3 = Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm1d(self.expansion * planes)
        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Res_Encoder(Module):
    def __init__(self, block=BasicBlock, num_blocks=[3, 4, 6, 3], in_channel=21, strategy='mean',
                 zero_init_residual=False):
        super(Res_Encoder, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv1d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
        self.bn1 = BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, num_blocks[3], stride=2)
        self.layer6 = self._make_layer(block, 2048, num_blocks[3], stride=2)
        self.avgpool = AdaptiveAvgPool1d(1)
        self.maxpool = AdaptiveMaxPool1d(1)
        self.fc = Sequential(
            Linear(2048, 1024),
            Linear(1024, 512),
            Linear(512, 256),
            Linear(256, 2)
        )
        self.Loss = CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm1d, GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        if strategy == 'mean':
            self.fusion = mean
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # out = self.avgpool(out)
        # out = self.maxpool(out)
        return out


class Res_Decoder(Module):
    def __init__(self, with_last_relu=False):
        super(Res_Decoder, self).__init__()
        self.fc = Linear(1, 32)
        self.block0 = UpSample(2048)
        self.block1 = ConvBlock(2048, 1024, 1024)
        self.upblock1 = UpSample(1024)
        self.block2 = ConvBlock(1024, 512, 512)
        self.upblock2 = UpSample(512)
        self.block3 = ConvBlock(512, 256, 256)
        self.upblock3 = UpSample(256)
        self.block4 = ConvBlock(256, 128, 128)
        self.upblock4 = UpSample(128)
        self.block5 = ConvBlock(128, 64, 64)
        self.upblock5 = UpSample(64)
        self.block6 = ConvBlock(64, 32, 21, with_last_relu)

    def forward(self, x):
        # x = x.repeat(1, 1, 32)
        # x = self.fc(x)
        # x = self.block0(x)
        # x = self.block0(x)
        # x = self.block0(x)
        # x = self.block0(x)
        # x = self.block0(x)
        x = self.upblock1(self.block1(x))
        x = self.upblock2(self.block2(x))
        x = self.upblock3(self.block3(x))
        x = self.upblock4(self.block4(x))
        x = self.upblock5(self.block5(x))
        x = self.block6(x)
        return x


class Resnet_Encoder(Module):
    def __init__(self, with_last_relu=False):
        super(Resnet_Encoder, self).__init__()
        self.encoder = Res_Encoder()
        self.decoder = Res_Decoder(with_last_relu)
        self.Loss = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, pred, true):
        reconloss = self.Loss(pred, true)
        return dict(Recon_Loss=reconloss, Total_Loss=reconloss)


# ---------------------------------------Resnet_Encoder------------------------------------


# ---------------------------------------Resnet_Mem_Encoder-----------------------------------
class Res_Mem_Encoder(Module):
    def __init__(self, block=BasicBlock, num_blocks=[3, 4, 6, 3], in_channel=21, strategy='mean',
                 zero_init_residual=False):
        super(Res_Mem_Encoder, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv1d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
        self.bn1 = BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 1024, num_blocks[3], stride=2)
        self.layer6 = self._make_layer(block, 2048, num_blocks[3], stride=2)
        self.avgpool = AdaptiveAvgPool1d(1)
        self.fc = Sequential(
            Linear(2048, 1024),
            Linear(1024, 512),
            Linear(512, 256),
            Linear(256, 2)
        )
        self.Loss = CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm1d, GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        if strategy == 'mean':
            self.fusion = mean
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.avgpool(out)
        return out


class Res_Mem_Decoder(Module):
    def __init__(self, with_last_relu=False):
        super(Res_Mem_Decoder, self).__init__()
        self.block1 = ConvBlock(2048, 1024, 1024)
        self.upblock1 = UpSample(1024)
        self.block2 = ConvBlock(1024, 512, 512)
        self.upblock2 = UpSample(512)
        self.block3 = ConvBlock(512, 256, 256)
        self.upblock3 = UpSample(256)
        self.block4 = ConvBlock(256, 128, 128)
        self.upblock4 = UpSample(128)
        self.block5 = ConvBlock(128, 64, 64)
        self.upblock5 = UpSample(64)
        self.block6 = ConvBlock(64, 32, 21, with_last_relu)

    def forward(self, x):
        x = x.repeat(1, 1, 32)
        x = self.upblock1(self.block1(x))
        x = self.upblock2(self.block2(x))
        x = self.upblock3(self.block3(x))
        x = self.upblock4(self.block4(x))
        x = self.upblock5(self.block5(x))
        x = self.block6(x)
        return x


class Resnet_MemEncoder(Module):
    def __init__(self, num_protos=30, fea_dim=2048, with_last_relu=False):
        super(Resnet_MemEncoder, self).__init__()
        self.encoder = Res_Mem_Encoder()
        self.decoder = Res_Mem_Decoder(with_last_relu)
        self.mem = MemModule(num_protos, fea_dim)
        self.Loss = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        y = self.mem(x)
        # print('similarity', y['similarity'])
        # print(y['similarity'].shape)
        # sim = y['similarity']
        # for i in range(sim.size(0)):
        #     sum = torch.sum(sim[i]>=0.015)
        #     print('sum', sum)
        z = self.decoder(y['output'])
        return z, y['output'], y['similarity'], x

    def loss(self, pred, label):
        pred_recon = pred[0]
        x_mem = pred[1]
        similarity = pred[2]
        x_code = pred[3]
        truedata = label
        align_loss = self.Loss(x_mem, x_code)
        recon_loss = self.Loss(pred_recon, truedata)
        entropy_loss = self.EntropyLoss(similarity)
        total_loss = recon_loss + 0.002 * entropy_loss + align_loss
        return dict(Recon_Loss=recon_loss, Entropy_Loss=entropy_loss, Align_Loss=align_loss, Total_Loss=total_loss)

    def EntropyLoss(self, similarity, eps=1e-12):
        similarity = similarity.permute(0, 2, 1)  # [B, N, T] -> [B, T, N]
        similarity = similarity.contiguous().view(-1, similarity.shape[-1])  # [BT, N]
        res = similarity * log(similarity + eps)  # [BT, N]
        res = -1.0 * res.sum(dim=1)  # [BT]
        loss = res.mean()
        return loss


# ---------------------------------------Resnet_Mem_Encoder-----------------------------------


# ------------------------------------SkipAutoEncoder--------------------------------------
class SkipBlock(Module):
    def __init__(self, in_channel, out_channel):
        super(SkipBlock, self).__init__()
        self.conv1 = Conv1d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = BatchNorm1d(out_channel)
        self.upsample = UpSample(out_channel)
        self.conv2 = Conv1d(in_channel, out_channel, 3, 1, 1)
        self.bn2 = BatchNorm1d(out_channel)
        self.relu = LeakyReLU(inplace=True)

    def forward(self, x, skip):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        x = cat((x, skip), 1)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SkipDecoder(Module):
    def __init__(self, with_last_relu=False):
        super(SkipDecoder, self).__init__()
        self.skipblock1 = SkipBlock(512, 256)
        self.skipblock2 = SkipBlock(256, 128)
        self.skipblock3 = SkipBlock(128, 64)
        self.skipblock4 = SkipBlock(64, 32)
        self.convblock = ConvBlock(32, 10, 10, with_last_relu)

    def forward(self, x, x4, x3, x2, x1):
        # x4 [512, 14] x3 [256, 28]  x2 [128, 56]  x1 [64, 112]
        x = self.skipblock1(x, x4)
        x = self.skipblock2(x, x3)
        x = self.skipblock3(x, x2)
        x = self.skipblock4(x, x1)
        x = self.convblock(x)  # [64, 112] -> [3, 112]
        return x


class SkipAutoEncoder(Module):
    def __init__(self, with_last_relu=False):
        super(SkipAutoEncoder, self).__init__()
        self.encoder = Encoder(True)
        self.decoder = SkipDecoder(with_last_relu)
        self.Loss = MSELoss()

    def forward(self, x):
        x, x4, x3, x2, x1 = self.encoder(x)
        x = self.decoder(x, x4, x3, x2, x1)
        return x

    def loss(self, pred, true):
        reconloss = self.Loss(pred, true)
        return dict(Recon_Loss=reconloss, Total_Loss=reconloss)


# ------------------------------------SkipAutoEncoder--------------------------------------

# ------------------------------------MemAutoEncoder--------------------------------------
class MemModule(Module):
    def __init__(self, num_protos=20, fea_dim=512, shrink_thres=0.015):
        super(MemModule, self).__init__()
        self.num_protos = num_protos
        self.fea_dim = fea_dim
        self.proto = Parameter(Tensor(self.num_protos, self.fea_dim))  # N * 512
        self.shrink_thres = shrink_thres
        self.initial_weight()

    def initial_weight(self):
        stdv = 1. / sqrt(self.fea_dim)
        self.proto.data.uniform_(-stdv, stdv)

    def hard_shrink_relu(self, input, lambd=0.0, epsilon=1e-12):
        output = (relu(input - lambd) * input) / (abs(input - lambd) + epsilon)
        return output

    def forward(self, X):
        # print(X.shape)
        batch, channel, T = X.shape  # [B, 512, T]
        X = X.permute(0, 2, 1)  # [B, T, 512]
        X = X.contiguous().view(-1, channel)  # [BT, 512]
        similarity = linear(X, self.proto)  # [BT, 512] * [512, N] = [BT, N]
        similarity = softmax(similarity, dim=1)  # [BT, N]
        if self.shrink_thres > 0:
            similarity = self.hard_shrink_relu(similarity, lambd=self.shrink_thres)
            similarity = normalize(similarity, p=1, dim=1)  # [BT, N]
        mem_trans = self.proto.permute(1, 0)  # [512, N]
        proto_based_fea = linear(similarity, mem_trans)  # [BT, N] *  [N, 512] = [BT, 512]
        proto_based_fea = proto_based_fea.view(batch, T, channel)  # [B, T, 512]
        proto_based_fea = proto_based_fea.permute(0, 2, 1)  # [B, 512, T]
        similarity = similarity.view(batch, T, self.num_protos)  # [B, T, N]
        similarity = similarity.permute(0, 2, 1)  # [B, N, T]
        return {'output': proto_based_fea, 'similarity': similarity}


class MemAutoEncoder(Module):
    def __init__(self, num_protos=20, fea_dim=512, with_last_relu=False):
        super(MemAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(with_last_relu)
        self.mem = MemModule(num_protos, fea_dim)
        self.Loss = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        y = self.mem(x)
        z = self.decoder(y['output'])
        return z, y['output'], y['similarity'], x

    def loss(self, pred, label):
        pred_recon = pred[0]
        x_mem = pred[1]
        similarity = pred[2]
        x_code = pred[3]
        truedata = label
        align_loss = self.Loss(x_mem, x_code)
        recon_loss = self.Loss(pred_recon, truedata)
        entropy_loss = self.EntropyLoss(similarity)
        total_loss = recon_loss + 0.002 * entropy_loss + align_loss
        return dict(Recon_Loss=recon_loss, Entropy_Loss=entropy_loss, Align_Loss=align_loss, Total_Loss=total_loss)

    def EntropyLoss(self, similarity, eps=1e-12):
        similarity = similarity.permute(0, 2, 1)  # [B, N, T] -> [B, T, N]
        similarity = similarity.contiguous().view(-1, similarity.shape[-1])  # [BT, N]
        res = similarity * log(similarity + eps)  # [BT, N]
        res = -1.0 * res.sum(dim=1)  # [BT]
        loss = res.mean()
        return loss


# ------------------------------------MemAutoEncoder--------------------------------------


# ------------------------------------VAEEncoder------------------------------------------
def reparameterize(mu, logvar):
    std = exp(0.5 * logvar)
    eps = randn_like(std)
    return eps * std + mu


class VAEEncoder(Module):
    def __init__(self, dim=1000):
        super(VAEEncoder, self).__init__()
        self.conv1 = Conv1d(10, 8, 4, 2, 1)
        self.bn1 = BatchNorm1d(8)
        self.conv2 = Conv1d(8, 16, 4, 2, 1)
        self.bn2 = BatchNorm1d(16)
        self.act = LeakyReLU()
        self.fc = Linear(16 * dim // 4, 1000)
        self.fc_mean = Linear(1000, 10)
        self.fc_var = Linear(1000, 10)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = flatten(x, start_dim=1)
        x = self.act(self.fc(x))
        mean = self.fc_mean(x)
        var = self.fc_var(x)
        return mean, var


class VAEDecoder(Module):
    def __init__(self, dim):
        super(VAEDecoder, self).__init__()
        self.dim = dim // 4
        self.fc1 = Linear(10, 1000)
        self.fc2 = Linear(1000, 16 * dim // 4)
        self.conv1 = ConvTranspose1d(16, 8, 4, 2, 1)
        self.bn1 = BatchNorm1d(8)
        self.conv2 = ConvTranspose1d(8, 10, 4, 2, 1)
        self.act = LeakyReLU()
        self.dropout = Dropout(0.3)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        x = x.view(-1, 16, self.dim)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return sigmoid(x)


class VAE(Module):
    def __init__(self, dim=1000, num_z=5, reductions='mean'):
        super(VAE, self).__init__()
        self.num_z = num_z
        self.encoder = VAEEncoder(dim)
        self.decoder = VAEDecoder(dim)
        self.Loss = L1Loss(reduction=reductions)

    def forward(self, x):
        Z = []
        mean, var = self.encoder(x)
        for i in range(self.num_z):
            z = (reparameterize(mean, var))
            Z.append(self.decoder(z))
        return Z, mean, var

    def loss(self, X, turelabel):
        recon_Loss = 0
        Z, mean, var = X
        for i in range(self.num_z):
            recon_Loss += self.Loss(Z[i], turelabel)
        recon_Loss /= self.num_z
        kld_Loss = 0.8 * mean(-0.5 * sum(1 + var - mean ** 2 - var.exp(), dim=1), dim=0)
        total_Loss = recon_Loss + kld_Loss
        return dict(Recon_Loss=recon_Loss, KL_Loss=kld_Loss, Total_Loss=total_Loss)


# ------------------------------------VAEEncoder------------------------------------------

# ------------------------------------EstimatorAutoEncoder--------------------------------
class Estimator1D(Module):
    def __init__(self, code_length, fm_list, cpd_dim):
        """
        Class constructor.
        :param code_length: the dimensionality of latent vectors.
        :param fm_list: list of channels for each MFC layer.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(Estimator1D, self).__init__()

        self.code_length = code_length
        self.fm_list = fm_list
        self.cpd_dim = cpd_dim

        activation_fn = LeakyReLU()

        # Add autoregressive layers
        layers_list = []
        mask_type = 'A'
        fm_in = 1
        for l in range(0, len(fm_list)):
            fm_out = fm_list[l]
            layers_list.append(
                MaskedFullyConnection(mask_type=mask_type,
                                      in_features=fm_in * code_length,
                                      out_features=fm_out * code_length,
                                      in_channels=fm_in, out_channels=fm_out)
            )
            layers_list.append(activation_fn)

            mask_type = 'B'
            fm_in = fm_list[l]

        # Add final layer providing cpd params
        layers_list.append(
            MaskedFullyConnection(mask_type=mask_type,
                                  in_features=fm_in * code_length,
                                  out_features=cpd_dim * code_length,
                                  in_channels=fm_in,
                                  out_channels=cpd_dim))

        self.layers = Sequential(*layers_list)

    def forward(self, x):
        h = unsqueeze(x, dim=1)  # add singleton channel dim
        h = self.layers(h)
        o = h
        return o


class BaseModule(Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """

    def load_w(self, checkpoint_path):
        # type: (str) -> None
        """
        Loads a checkpoint into the state_dict.
        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(load(checkpoint_path))

    def __repr__(self):
        # type: () -> str
        """
        String representation
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        # type: () -> int
        """
        Number of parameters of the model.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)


class MaskedFullyConnection(BaseModule, Linear):
    def __init__(self, mask_type, in_channels, out_channels, *args, **kwargs):
        self.mask_type = mask_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(MaskedFullyConnection, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())

        # Build mask
        self.mask.fill_(0)
        for f in range(0 if mask_type == 'B' else 1, self.out_features // self.out_channels):
            start_row = f * self.out_channels
            end_row = (f + 1) * self.out_channels
            start_col = 0
            end_col = f * self.in_channels if mask_type == 'A' else (f + 1) * self.in_channels
            if start_col != end_col:
                self.mask[start_row:end_row, start_col:end_col] = 1

        self.weight.mask = self.mask

    def forward(self, x):
        # Reshape
        x = transpose(x, 1, 2).contiguous()
        x = x.view(len(x), -1)

        # Mask weights and call fully connection
        self.weight.data *= self.mask
        o = super(MaskedFullyConnection, self).forward(x)

        # Reshape again
        o = o.view(len(o), -1, self.out_channels)
        o = transpose(o, 1, 2).contiguous()
        return o

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'mask_type=' + str(self.mask_type) \
            + ', in_features=' + str(self.in_features // self.in_channels) \
            + ', out_features=' + str(self.out_features // self.out_channels) \
            + ', in_channels=' + str(self.in_channels) \
            + ', out_channels=' + str(self.out_channels) \
            + ', n_params=' + str(self.n_parameters) + ')'


class EstimatorAutoEncoder(Module):
    def __init__(self, code_length=512, cpd_dim=128, with_last_relu=False):
        super(EstimatorAutoEncoder, self).__init__()
        self.cpd_dim = cpd_dim
        self.encoder = Encoder()
        activation_fn = LeakyReLU()
        deepest_shape = [512, 64]
        self.encoderfc = Sequential(
            Linear(in_features=reduce(mul, deepest_shape), out_features=2048),
            BatchNorm1d(num_features=2048),
            activation_fn,
            Linear(in_features=2048, out_features=code_length),
            Sigmoid()
        )
        self.decoder = Decoder(with_last_relu)
        self.decoderfc = Sequential(
            Linear(in_features=code_length, out_features=2048),
            BatchNorm1d(num_features=2048),
            activation_fn,
            Linear(in_features=2048, out_features=reduce(mul, deepest_shape)),
            BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )
        self.estimator = Estimator1D(code_length, [8, 8, 8, 8], cpd_dim)
        self.Loss = MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(len(z), -1)
        z = self.encoderfc(z)
        z_dist = self.estimator(z)
        x_r = self.decoderfc(z)
        x_r = x_r.view(len(x_r), 512, 64)
        x_r = self.decoder(x_r)
        return x_r, z, z_dist

    def loss(self, pred, label):
        recon = pred[0]
        x_code = pred[1]
        esti_value = pred[2]
        truedata = label
        recon_loss = self.Loss(recon, truedata)
        autoregressionloss = 0.0005 * self.Autoregressionloss(x_code, esti_value)  # 0.0005,1,0.001
        total_loss = recon_loss + autoregressionloss
        return dict(Recon_Loss=recon_loss, Autoregression_Loss=autoregressionloss, Total_Loss=total_loss)

    def Autoregressionloss(self, z, z_dist, eps=finfo(float).eps):
        z_d = z.detach()

        # Apply softmax
        z_dist = softmax(z_dist, dim=1)

        # Flatten out codes and distributions
        z_d = z_d.view(len(z_d), -1).contiguous()
        z_dist = z_dist.view(len(z_d), self.cpd_dim, -1).contiguous()

        # Log (regularized), pick the right ones
        z_dist = clamp(z_dist, eps, 1 - eps)
        log_z_dist = log(z_dist)
        index = clamp(unsqueeze(z_d, dim=1) * self.cpd_dim, min=0,
                      max=(self.cpd_dim - 1)).long()
        selected = gather(log_z_dist, dim=1, index=index)
        selected = squeeze(selected, dim=1)

        # Sum and mean
        S = sum(selected, dim=-1)
        nll = - mean(S)

        return nll


# ------------------------------------EstimatorAutoEncoder--------------------------------

# ------------------------------------EstimatorMemAutoEncoder-----------------------------
class EstimatorMemAutoEncoder(Module):
    def __init__(self, code_length=512, cpd_dim=128, num_protos=20, fea_dim=512, with_last_relu=False):
        super(EstimatorMemAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.cpd_dim = cpd_dim
        activation_fn = LeakyReLU()
        deepest_shape = [512, 64]
        # self.encoderfc = nn.Sequential(
        #     nn.Linear(in_features=reduce(mul, deepest_shape), out_features=2048),
        #     nn.BatchNorm1d(num_features=2048),
        #     activation_fn,
        #     nn.Linear(in_features=2048, out_features=code_length),
        #     nn.Sigmoid()
        # )
        self.decoder = Decoder(with_last_relu)
        # self.decoderfc = nn.Sequential(
        #     nn.Linear(in_features=code_length, out_features=2048),
        #     nn.BatchNorm1d(num_features=2048),
        #     activation_fn,
        #     nn.Linear(in_features=2048, out_features=reduce(mul, deepest_shape)),
        #     nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
        #     activation_fn
        # )
        self.mem = MemModule(num_protos, fea_dim)
        self.estimator = Estimator1D(code_length, [4, 4, 4, 4], cpd_dim)
        self.Loss = MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        z = self.mem(x)
        y1 = mean(x, -1)
        y2 = mean(z['output'], -1)
        esti_value = self.estimator(y1 - y2)
        pred = self.decoder(z['output'])
        return pred, z['output'], z['similarity'], x, esti_value, y1 - y2

    def loss(self, pred, label):
        pred_recon = pred[0]
        x_mem = pred[1]
        similarity = pred[2]
        x_code = pred[3]
        esti_value = pred[4]
        y1 = pred[5]
        truedata = label
        align_loss = self.Loss(x_mem, x_code)
        recon_loss = self.Loss(pred_recon, truedata)
        entropy_loss = self.EntropyLoss(similarity)
        autoregressionloss = 0.0005 * self.Autoregressionloss(y1, esti_value)
        total_loss = recon_loss + 0.002 * entropy_loss + align_loss + autoregressionloss
        return dict(Recon_Loss=recon_loss, Entropy_Loss=entropy_loss,
                    Autoregression_Loss=autoregressionloss,
                    Align_Loss=align_loss, Total_Loss=total_loss)

    def EntropyLoss(self, similarity, eps=1e-12):
        similarity = similarity.permute(0, 2, 1)  # [B, N, T] -> [B, T, N]
        similarity = similarity.contiguous().view(-1, similarity.shape[-1])  # [BT, N]
        res = similarity * log(similarity + eps)  # [BT, N]
        res = -1.0 * res.sum(dim=1)  # [BT]
        loss = res.mean()
        return loss

    def Autoregressionloss(self, z, z_dist, eps=finfo(float).eps):
        z_d = z.detach()

        # Apply softmax
        z_dist = softmax(z_dist, dim=1)

        # Flatten out codes and distributions
        z_d = z_d.view(len(z_d), -1).contiguous()
        z_dist = z_dist.view(len(z_d), self.cpd_dim, -1).contiguous()

        # Log (regularized), pick the right ones
        z_dist = clamp(z_dist, eps, 1 - eps)
        log_z_dist = log(z_dist)
        index = clamp(unsqueeze(z_d, dim=1) * self.cpd_dim, min=0,
                      max=(self.cpd_dim - 1)).long()
        selected = gather(log_z_dist, dim=1, index=index)
        selected = squeeze(selected, dim=1)

        # Sum and mean
        S = sum(selected, dim=-1)
        nll = - mean(S)

        return nll
# ------------------------------------EstimatorMemAutoEncoder-----------------------------
#
#
#
#
#
# if __name__ == '__main__':
#     # input = torch.rand(64, 21, 1536).float().cuda()
#     # model = VAE(1536).cuda().float()
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     input = torch.rand(8,10, 1000).float().cuda()
#     model = VAEDecoder(100).float().cuda()
#     dec = model(input)
#     out = model.loss(dec, input)
#     print(out)
