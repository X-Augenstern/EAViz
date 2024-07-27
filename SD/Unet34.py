from torch.nn import init
from torch import nn
from SD.Config import loss_weight
from SD.resnet import resnet34


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_name.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_name.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(Unet34):
    Unet34.apply(weights_init_kaiming)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, n=1, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv1d(in_size, out_size, kernel_size=ks, stride=s, padding=p),
                                 nn.BatchNorm1d(out_size),
                                 nn.ReLU(inplace=True))
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size
        for m in self.children():
            init_weights(m)

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class Unet34(nn.Module):
    def __init__(self, n_channels=19, n_classes=2, SA=False):
        super(Unet34, self).__init__()
        self.Loss = nn.CrossEntropyLoss()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.SA = SA
        self.filters = filters = [32, 64, 128, 256, 512]  # 也是5层 原本的模型使用的是64-1024
        self.backbone = resnet34(initChannel=n_channels, SA=SA)
        self.stageclassifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.Linear(128, 3)
        )
        self.CatChannels = filters[0]  # 拼接通道数 32  高的maxpooling 低 upsample
        self.CatBlocks = 5  # 5次拼接
        self.UpChannels = self.CatChannels * self.CatBlocks  # 上采样通道固定160
        self.up_down = nn.ModuleList()
        padding = [0, 1, 1, 1]
        for i in range(len(filters) - 1):
            pad = padding[i]
            now = filters[len(filters) - i - 1]
            next = filters[len(filters) - i - 2]
            self.up_down.append(nn.Sequential(
                nn.ConvTranspose1d(now, now, kernel_size=3, stride=2, padding=pad, output_padding=pad),
                unetConv2(now, next, ks=3, padding=1)))
        # 代替传统的插值方法实现上采样操作  特征上采样； 将低维特征映射到高位空间；
        self.outconv = nn.Sequential(nn.ConvTranspose1d(filters[0], filters[0], kernel_size=3, stride=3),
                                     nn.Conv1d(filters[0], self.n_classes, 3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init_weights(m)
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m)

    def forward(self, inputs):
        # ----------------Encoder------------------
        out = self.backbone(inputs)
        stage_feat = out['stage_feat']
        h1, h2, h3, h4, hd5 = out['seg_feat']
        # ----------------StageClassification-----
        d2 = self.stageclassifier(stage_feat)
        # ----------------Segementation----------
        x = hd5
        for up_down in self.up_down:
            x = up_down(x)
        # print('x',x.shape)
        # exit(0)
        d1 = self.outconv(x)
        # 1：语义分割输出  2：睡眠分期输出  3： 语义分割通道选择权重 4：睡眠分期通道选择权重
        return {'seg_out': d1,
                'stage_out': d2,
                'seg_att': out['seg_att'],
                'stage_att': out['stage_att'],
                'feat': out['stage_feat']}

    def loss(self, predict, seg_label, stage_label):
        seg, stage = predict['seg_out'], predict['stage_out']
        Reg_Loss = self.Loss(seg, seg_label)  # 使用的交叉熵，与原模型有区别
        CE_Loss = self.Loss(stage, stage_label)
        return dict(reg_Loss=Reg_Loss, CE_Loss=CE_Loss, total_Loss=loss_weight*Reg_Loss + (1-loss_weight)*CE_Loss)
#
#
# if __name__ == '__main__':
#     x = torch.randn(2, 19, 15000).cuda().float()  # b*c*signal out 12*2*15000
#     net = Unet34(n_channels=1, SA=True).cuda().float()
#     print(net)
#     out = net(x)
#     for i in out.values():
#         print(i.shape)
