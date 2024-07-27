import torch
from torch import nn
from torch.autograd import Variable
from SeiD.A3D_model import SpatioTemporalResLayer, SpatioTemporalResBlock, SpatioTemporalConv


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # https://pytorch.org/docs/master/nn.html?highlight=lstm#torch.nn.LSTM
        # 参考官方文档
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.pool = nn.MaxPool2d((100, 2))
        # self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device))
        out, _ = self.lstm(x, (h0, c0))
        # 选择最后一个时间点的output

        out = self.pool(out)
        b, c, d = out.size()
        out = out.reshape(b, 1, c, d)

        return out


class Net3Dto2D(nn.Module):
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(Net3Dto2D, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 16, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 1])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.blk1_3D = SpatioTemporalResLayer(16, 16, 3, 21, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.blk2_3D = SpatioTemporalResLayer(16, 32, 3, 11, layer_sizes[1], block_type=block_type, downsample=True)
        self.blk3_3D = SpatioTemporalResLayer(32, 64, 3, 6, layer_sizes[2], block_type=block_type, downsample=True)
        self.blk4_3D = SpatioTemporalResLayer(64, 128, 3, 3, layer_sizes[3], block_type=block_type, downsample=True)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),  # inplace = True原地操作
        )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),  # inplace = True原地操作
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),  # inplace = True原地操作
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),  # inplace = True原地操作
        )

        # followed 4 blocks
        # [b, 16, h, w] => [b, 32, h ,w]
        # global average pooling of the output
        self.pool1 = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, y):
        # new addition
        feature_map = []

        x = self.conv1(x)
        feature_map.append(x)
        x = self.maxpool(x)
        # print(y.shape)

        y = self.cnn(y)
        x1 = self.blk1_3D(x, y)
        feature_map.append(x1)

        y = self.cnn1(y)
        x2 = self.blk2_3D(x1, y)
        feature_map.append(x2)

        y = self.cnn2(y)
        x3 = self.blk3_3D(x2, y)
        feature_map.append(x3)

        y = self.cnn3(y)
        x4 = self.blk4_3D(x3, y)
        feature_map.append(x4)

        x = self.pool1(x4)

        return x.view(-1, 128), y.view(y.size(0), -1), feature_map


class Classifier1D_2D_3D(nn.Module):
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(Classifier1D_2D_3D, self).__init__()

        self.R1 = RNN(21, 42, 2, 11)
        self.R2 = Net3Dto2D((2, 2, 2, 2))
        # #sog门
        # self.linear11 = torch.nn.Sequential(nn.Linear(143, 128),nn.Dropout(p=0.5))
        # self.linear12 = nn.Linear(128, 64)
        # self.linear13 = nn.Linear(64, 32)
        # self.linear21 = torch.nn.Sequential(nn.Linear(143, 128),nn.Dropout(p=0.5))
        # self.linear22 = nn.Linear(128, 64)
        # self.linear23 = nn.Linear(64, 32)
        # self.linear14 = nn.Linear(32, 6)
        # self.linear24 = nn.Linear(32, 2)
        # #sog门

        # rsog门
        self.linear11 = torch.nn.Sequential(nn.Linear(143, 128), nn.Dropout(p=0.5))
        self.linear12 = nn.Linear(128, 64)
        self.linear13 = nn.Linear(64, 32)
        self.linear21 = torch.nn.Sequential(nn.Linear(143, 128), nn.Dropout(p=0.5))
        self.linear22 = nn.Linear(128, 64)
        self.linear23 = nn.Linear(64, 32)
        self.linear14 = nn.Linear(32, 6)
        self.linear24 = nn.Linear(32, 2)
        self.relu11 = nn.ReLU()
        self.relu12 = nn.ReLU()
        self.relu13 = nn.ReLU()
        self.relu21 = nn.ReLU()
        self.relu22 = nn.ReLU()
        self.relu23 = nn.ReLU()
        # rsog门

        # 不加门
        # self.age_fc_layers1 = torch.nn.Sequential(
        #     nn.Linear(143, 6),
        #
        #     # torch.nn.Sigmoid()
        # )
        # self.gender_fc_layers1 = torch.nn.Sequential(
        #     nn.Linear(143, 2),
        #
        # )
        # 不加门
        # self.__init_weight()
        #
        # if pretrained:
        #     self.__load_pretrained_weights()

    def forward(self, x, y):  # 原始训练网络
        x = self.R1(x)  # x(8, 4000, 21) y(8, 3, 21, 32, 32)
        out1, out2, feature_map = self.R2(y, x)
        out = torch.cat((out1, out2), 1)
        # sog门
        # out_age1 = self.linear11(out)
        # out_gender1 = self.linear21(out)
        # h1=nn.Softmax(dim=1)(out_gender1)*out_gender1+out_age1
        # h2=nn.Softmax(dim=1)(out_age1)*out_age1+out_gender1
        # out_age2 = self.linear12(h1)
        # out_gender2 = self.linear22(h2)
        # h3 = nn.Softmax(dim=1)(out_gender2) * out_gender2 + out_age2
        # h4 = nn.Softmax(dim=1)(out_age2) * out_age2 + out_gender2
        # out_age3 = self.linear13(h3)
        # out_gender3 = self.linear23(h4)
        # h5 = nn.Softmax(dim=1)(out_gender3) * out_gender3 + out_age3
        # h6 = nn.Softmax(dim=1)(out_age3) * out_age3 + out_gender3
        # out_age = self.linear14(h5)
        # out_gender = self.linear24(h6)
        # sog门

        # #rsog门
        out_age1 = self.linear11(out)
        out_gender1 = self.linear21(out)
        h1 = nn.Softmax(dim=1)(self.relu21(out_gender1)) * out_gender1 + out_age1
        h2 = nn.Softmax(dim=1)(self.relu11(out_age1)) * out_age1 + out_gender1
        out_age2 = self.linear12(h1)
        out_gender2 = self.linear22(h2)
        h3 = nn.Softmax(dim=1)(self.relu22(out_gender2)) * out_gender2 + out_age2
        h4 = nn.Softmax(dim=1)(self.relu12(out_age2)) * out_age2 + out_gender2
        out_age3 = self.linear13(h3)
        out_gender3 = self.linear23(h4)
        h5 = nn.Softmax(dim=1)(self.relu23(out_gender3)) * out_gender3 + out_age3
        h6 = nn.Softmax(dim=1)(self.relu13(out_age3)) * out_age3 + out_gender3
        out_age = self.linear14(h5)
        out_gender = self.linear24(h6)
        # #rsog门

        # 不加门
        # out_age = self.age_fc_layers1(out)
        # out_gender = self.gender_fc_layers1(out)
        # 不加门
        return out_age, out_gender, feature_map
