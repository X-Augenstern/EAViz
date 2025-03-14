from torch import empty, device, cuda, float16, from_numpy, load, zeros, cat, float as t_float
from numpy import ndarray
from torch.nn.functional import avg_pool3d
from torch.nn import Module, Hardswish, LeakyReLU, ReLU, ReLU6, SiLU, Upsample, ModuleList, Conv3d, BatchNorm3d, \
    MaxPool3d, AdaptiveAvgPool3d, Linear, init, Sequential
from functools import partial


# ModuleNotFoundError: No module named 'models'
# 在使用torch.load()加载模型的文件最开始加上
# path.insert(0, path1.abspath('./VD'))


def loadModel(detectWeights, actionWeights, cfg):
    model1 = DetectModule(detectWeights, device=cfg.device, fp16=cfg.half)
    img_detection = empty(*cfg.warmshape_for_object_detection, dtype=t_float, device=cfg.device)
    model1.forward(img_detection)

    model2 = RecognitionModule(actionWeights, cfg.device)
    img_action = empty(*cfg.warmshape_for_action_recognition, dtype=t_float, device=cfg.device)
    model2.forward(img_action)
    return model1, model2


class DetectModule(Module):
    def __init__(self, weights='./weights/best.pt', device=device('cuda:0'), fp16=False, fuse=True):
        super().__init__()
        stride = 32  # default stride
        model = attempt_load(weights, device=device, inplace=True, fuse=fuse)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        if self.fp16 and im.dtype != float16:
            im = im.half()
        y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return from_numpy(x).to(self.device) if isinstance(x, ndarray) else x


def attempt_load(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    ckpt = load(weights, map_location='cpu')  # load
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
    model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode
    # model.append(ckpt.fuse() if fuse and hasattr(ckpt, 'fuse') else ckpt)  # model in eval mode
    for m in model.modules():
        t = type(m)
        if t in (Hardswish, LeakyReLU, ReLU, ReLU6, SiLU) or '\'models.yolo.Detect\'' in str(t):
            m.inplace = inplace
            if '\'models.yolo.Detect\'' in str(t) and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [zeros(1)] * m.nl)
        elif t is Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    return model[-1]


class Ensemble(ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x=None, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = cat(y, 1)  # nms ensemble
        return y, None


def RecognitionModule(weight_path, device, numclasses=2):
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), n_classes=numclasses)
    weight = load(weight_path)['state_dict']
    model.load_state_dict(weight)
    model = model.to(device).float().eval()
    return model


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return Conv3d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return Conv3d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm3d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm3d(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 needC=True):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.needC = needC
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = Conv3d(n_input_channels,
                            self.in_planes,
                            kernel_size=(conv1_t_size, 7, 7),
                            stride=(conv1_t_stride, 2, 2),
                            padding=(conv1_t_size // 2, 3, 3),
                            bias=False)
        self.bn1 = BatchNorm3d(self.in_planes)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = AdaptiveAvgPool3d((1, 1, 1))
        self.fc = Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, Conv3d):
                init.kaiming_normal_(m.weight,
                                     mode='fan_out',
                                     nonlinearity='relu')
            elif isinstance(m, BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = zeros(out.size(0), planes - out.size(1), out.size(2),
                          out.size(3), out.size(4))
        if isinstance(out.data, cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.needC:
            x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x
