from time import time
from cv2 import rectangle, LINE_AA, getTextSize, putText
from numpy import asarray, stack, copy
from torch import zeros, cat, Tensor, mm, min as t_min, max as t_max, tensor, no_grad, argmax, from_numpy
from torchvision.ops import nms


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                        multi_label=False, labels=(), max_det=300, nm=0):
    if isinstance(prediction, (list, tuple)):  # in validation model output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time()
    mi = 5 + nc  # mask start index
    output = [zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = cat((x, v), 0)

        if not x.shape[0]:
            output = None
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes

        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]
        if (time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, Tensor) else copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (t_min(a2, b2) - t_max(a1, b1)).clamp(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def annotating_box(im, box, label='', color=(128, 128, 128),
                   txt_color=(255, 255, 255), line_width=None):
    assert im.data.contiguous, \
        'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
    lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    rectangle(im, p1, p2, color, thickness=lw, lineType=LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        rectangle(im, p1, p2, color, -1, LINE_AA)  # filled
        putText(im,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=LINE_AA)
    return asarray(im)


@no_grad()
def actionRecognition(actionModel, imgs, device):
    try:
        imgs = stack(imgs)
        imgs = imgs.transpose(3, 0, 1, 2)
        imgs = from_numpy(imgs).to(device).float().unsqueeze(0)
        res = actionModel(imgs)
        patientstate = argmax(res).cpu()
        return 'Seizure' if patientstate == 1 else 'Interictal'
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'None'


class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def detect(img, img0, last_box, model, cfg):
    pred, featuremap = model(img, augment=False, visualize=False)
    pred = non_max_suppression([pred[0], pred[2]], cfg.conf_thres, cfg.iou_thres,
                               cfg.classes, cfg.agnostic_nms, max_det=cfg.max_det)

    if pred is not None:
        pred = sorted(pred, key=lambda x: x[0][4], reverse=True)[0]
        last_box = pred.clone()
    elif last_box is not None:
        pred = last_box.clone()
    else:
        return None, None

    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
    *xyxy, conf, cls = pred[0]
    return last_box, xyxy


"""
    dataset prepareVid的实例化对象
    model1  目标检测模型
    model2  动作识别模型
    last_box上一帧检测出来的框
    cfg     配置文件
    color   Colors的实例化对象
"""

# def vd_api(dataset, output_path, model1, model2, last_box, cfg, color):
#     features = []
#     cnt = 0
#     dataset = iter(dataset)
#
#     output_frames = []
#
#     while True:
#         if cnt >= 60:
#             break
#         cnt += 1
#
#         path, img, img0, vid_cap = next(dataset)
#         if img is None:
#             break
#         im = img0.copy()
#
#         last_box, xyxy = detect(img, img0, last_box, model1, cfg)
#         if last_box is None:
#             print('未检测到患者')
#             return None
#
#         res_frame = img0.copy()  # 创建一个副本用于绘制标注框
#         res1 = annotating_box(res_frame, xyxy, color=color(1, True), line_width=cfg.line_thickness)
#         output_frames.append(res1)
#
#     features.append(resize(im[int(xyxy[1]):int(xyxy[3]),
#                            int(xyxy[0]):int(xyxy[2])],
#                            (112, 112)))
#     res2 = actionRecognition(model2, features, cfg.device)
#
#     output_width, output_height = output_frames[0].shape[1], output_frames[0].shape[0]
#     fourcc = VideoWriter_fourcc(*"mp4v")
#     output_video = VideoWriter(output_path, fourcc, 20, (output_width, output_height))
#     for frame in output_frames:
#         output_video.write(frame)
#     output_video.release()
#
#     return res2, last_box
