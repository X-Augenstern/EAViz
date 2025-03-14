from time import time
from copy import deepcopy
from pathlib import Path
from torch import cat, sigmoid, cuda, empty, diag, sqrt, mm, zeros, arange, meshgrid, stack, __version__
from torch.nn import Module, Conv2d, BatchNorm2d, Identity, Parameter
from torch.nn.functional import interpolate, pad
from math import gcd, log, ceil
from pkg_resources import parse_version

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


# VD.utils.torch_utils
def time_sync():
    # PyTorch-accurate time
    if cuda.is_available():
        cuda.synchronize()
    return time()


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # Scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (ceil(x * ratio / gs) * gs for x in (h, w))
    return pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def model_info(model, verbose=False, imgsz=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
        im = empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    except Exception:
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    # LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def fuse_conv_and_bn(conv, bn):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = Conv2d(conv.in_channels,
                       conv.out_channels,
                       kernel_size=conv.kernel_size,
                       stride=conv.stride,
                       padding=conv.padding,
                       dilation=conv.dilation,
                       groups=conv.groups,
                       bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = diag(bn.weight.div(sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


# VD.utils.general

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING: ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    # if hard:
    #     assert result, emojis(s)  # assert min requirements met
    # if verbose and not result:
    #     LOGGER.warning(s)
    return result


# VD.models.common

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SiLU(Module):
    # SiLU activation https://arxiv.org/pdf/1606.08415.pdf
    @staticmethod
    def forward(x):
        return x * sigmoid(x)


class Conv(Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    act = SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = BatchNorm2d(c2)
        self.act = self.act if act is True else act if isinstance(act, Module) else Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=gcd(c1, c2), d=d, act=act)


class Detect(Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        # super().__init__()
        # self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        # self.nl = len(anchors)  # number of detection layers
        # self.na = len(anchors[0]) // 2  # number of anchors
        # self.grid = [torch.empty(1)] * self.nl  # init grid
        # self.anchor_grid = [torch.empty(1)] * self.nl  # init anchor grid
        # self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        pass

    def forward(self, x):
        z = []  # inference output
        logits_ = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                logits = x[i][..., 5:]
                y = x[i].clone()
                y[..., :5 + self.nc].sigmoid_()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, etc = y.split((2, 2, self.no - 4), 4)  # tensor_split((2, 4, 5), 4) if torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = cat((xy, wh, etc), 4)
                z.append(y.view(bs, -1, self.no))
                logits_.append(logits.view(bs, -1, self.no - 5))

        return x if self.training else (cat(z, 1), cat(logits_, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = arange(ny, device=d, dtype=t), arange(nx, device=d, dtype=t)
        yv, xv = meshgrid(y, x, indexing='ij') if torch_1_10 else meshgrid(y, x)  # torch>=0.7 compatibility
        grid = stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class BaseModel(Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        featuremap = None
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            # if m.i == 17:
            #     featuremap = x.clone()
            # feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x, featuremap

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            # LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
            print(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        # LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        print(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            # LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
            print(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        # LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        # super().__init__()
        # if isinstance(cfg, dict):
        #     self.yaml = cfg  # model dict
        # else:  # is *.yaml
        #     import yaml  # for torch hub
        #     self.yaml_file = Path(cfg).name
        #     with open(cfg, encoding='ascii', errors='ignore') as f:
        #         self.yaml = yaml.safe_load(f)  # model dict
        #
        # # Define model
        # ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        # if nc and nc != self.yaml['nc']:
        #     # LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        #     self.yaml['nc'] = nc  # override yaml value
        # if anchors:
        #     # LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
        #     self.yaml['anchors'] = round(anchors)  # override yaml value
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # self.inplace = self.yaml.get('inplace', True)
        #
        # # Build strides, anchors
        # m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):
        #     s = 256  # 2x min stride
        #     m.inplace = self.inplace
        #     forward = lambda x: self.forward(x)
        #     m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
        #     check_anchor_order(m)
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once
        #
        # # Init weights, biases
        # initialize_weights(self)
        # self.info()
        # # LOGGER.info('')
        pass

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += log(0.6 / (m.nc - 0.99999)) if cf is None else log(cf / cf.sum())  # cls
            mi.bias = Parameter(b.view(-1), requires_grad=True)

# Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility

# def parse_model(d, ch):  # model_dict, input_channels(3)
#     # Parse a YOLOv5 model.yaml dictionary
#     # LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
#     anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
#     if act:
#         Conv.act = eval(act)  # redefine default activation, i.e. Conv.act = nn.SiLU()
#         # LOGGER.info(f"{colorstr('activation:')} {act}")  # print
#     na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
#     no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
#
#     layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
#     for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
#         m = eval(m) if isinstance(m, str) else m  # eval strings
#         for j, a in enumerate(args):
#             with contextlib.suppress(NameError):
#                 args[j] = eval(a) if isinstance(a, str) else a  # eval strings
#         n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
#         if m in {
#             Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus, CrossConv,
#             BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
#             c1, c2 = ch[f], args[0]
#             if c2 != no:  # if not output
#                 c2 = make_divisible(c2 * gw, 8)
#
#             args = [c1, c2, *args[1:]]
#             if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
#                 args.insert(2, n)  # number of repeats
#                 n = 1
#         elif m is nn.BatchNorm2d:
#             args = [ch[f]]
#         elif m is Concat:
#             c2 = sum(ch[x] for x in f)
#         elif m in {Detect}:
#             args.append([ch[x] for x in f])
#             if isinstance(args[1], int):  # number of anchors
#                 args[1] = [list(range(args[1] * 2))] * len(f)
#         elif m is Contract:
#             c2 = ch[f] * args[0] ** 2
#         elif m is Expand:
#             c2 = ch[f] // args[0] ** 2
#         else:
#             c2 = ch[f]
#         m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
#         t = str(m)[8:-2].replace('__main__.', '')  # module type
#         np = sum(x.numel() for x in m_.parameters())  # number params
#         m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
#         # LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
#         save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
#         layers.append(m_)
#         if i == 0:
#             ch = []
#         ch.append(c2)
#     # print(layers)
#     return nn.Sequential(*layers), sorted(save)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='newyolov5s.yaml', help='model.yaml')
#     parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--profile', action='store_true', help='profile model speed')
#     parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
#     parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
#     opt = parser.parse_args()
#     opt.cfg = check_yaml(opt.cfg)  # check YAML
#     print_args(vars(opt))
#     device = select_device(opt.device)
#
#     # Create model
#     im = torch.rand(1, 3, 640, 640).to(device)
#     model = Model(opt.cfg).to(device)
#     out = model(im)

# # Options
# if opt.line_profile:  # profile layer by layer
#     model(im, profile=True)
#
# elif opt.profile:  # profile forward-backward
#     results = profile(input=im, ops=[model], n=3)
#
# elif opt.test:  # test all models
#     for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
#         try:
#             _ = Model(cfg)
#         except Exception as e:
#             print(f'Error in {cfg}: {e}')
#
# else:  # report fused model summary
#     model.fuse()
