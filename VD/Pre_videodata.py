from cv2 import VideoCapture, resize, INTER_LINEAR, copyMakeBorder, BORDER_CONSTANT, CAP_PROP_FRAME_WIDTH, \
    CAP_PROP_FRAME_HEIGHT
from numpy import ascontiguousarray, mod
from torch import device, from_numpy


class LoadVideos:
    def __init__(self, cfg, input_list, stride, next_video_callback=None):
        self.video_path_list = input_list  # 视频路径列表
        self.num_video = len(self.video_path_list)
        self.stride = stride  # 步幅值
        self.cfg = cfg
        self.device = device(self.cfg.device)
        self.cap = None
        self.video_changed_callback = next_video_callback
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return self.num_video

    def __next__(self):
        """
        for path, im, img0, self.cap in (instance) | path, img, img0, self.vid_cap = next(dataset)
        在 for 循环中进行迭代时，实际上会调用 __iter__() 方法来初始化迭代器，
        然后在每次迭代时调用 __next__() 方法来获取下一个视频的路径、图像等信息，
        直到没有更多的视频需要迭代，此时会引发 StopIteration 异常来结束迭代。
        """
        while self.count < self.num_video:
            path = self.video_path_list[self.count]
            if not self.cap or not self.cap.isOpened():
                self.cap = VideoCapture(path)  # get video
            ret_val, img0 = self.cap.read()  # 从当前打开的视频中读取一帧
            if ret_val:  # 成功读取帧
                return self.process_frame(img0, path)
            else:  # 当前视频读取完毕或无法读取，尝试下一个视频
                self.move_to_next_video(path)
        raise StopIteration  # 所有视频都已尝试且无法读取

    def process_frame(self, img0, path):
        im = letterbox(img0, self.cfg.imgsz, stride=self.stride, auto=self.cfg.auto)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = from_numpy(ascontiguousarray(im)).to(self.device).float() / 255  # 处理后的图像数据
        if len(im.shape) == 3:
            im = im[None]
        return path, im, img0, self.cap

    def move_to_next_video(self, path):
        if self.cap and self.cap.isOpened():
            video_size = (
                int(self.cap.get(CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(CAP_PROP_FRAME_HEIGHT))
            )
            self.cap.release()
        else:
            video_size = None

        self.count += 1

        if self.video_changed_callback:
            self.video_changed_callback(path, video_size)


# 将图像调整为指定的尺寸并进行填充
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, stride=32):
    # new_shape：新图像的目标尺寸  color：填充区域的颜色   auto：是否使用步幅进行填充的调整
    # scaleFill：是否拉伸图像以填充目标尺寸  scaleup：是否允许图像放大  stride：填充和调整大小时的步幅参数
    # Resize and pad image while meeting stride-multiple constraints

    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Scale ratio (new / old) # 计算图像尺寸的缩放比例，以便将图像调整为目标尺寸
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)  # 缩放比例不超过1.0，以便在不放大图像的情况下进行调整

    # Compute padding
    ratio = r, r  # width, height ratios  # 在保持图像纵横比的情况下进行缩放操作
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 计算了调整大小后的图像尺寸（不进行填充的情况下） round进行四舍五入
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding  # 计算了填充的宽度和高度

    if auto:  # minimum rectangle
        dw, dh = mod(dw, stride), mod(dh, stride)  # wh padding  # 使用步幅进行调整，使得填充后的图像的宽度和高度都是步幅的倍数
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios  # 直接将图像的尺寸调整为目标尺寸，不进行填充

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize  # 将shape中的高度和宽度位置互换
        im = resize(im, new_unpad, interpolation=INTER_LINEAR)  # 线性插值缩放
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算顶部、底部、左侧和右侧的填充像素数
    im = copyMakeBorder(im, top, bottom, left, right, BORDER_CONSTANT, value=color)  # add border # 在图像的周围创建一个边框，以填充图像
    return im, ratio, (dw, dh)
