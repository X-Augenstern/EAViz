from PyQt5.QtCore import QThread, pyqtSignal
from io import BytesIO
from utils.config import AddressConfig, ThemeColorConfig
from numpy import mean, std, var, ndarray, array, corrcoef
from torch import device, tensor, load, from_numpy
from cv2 import CAP_PROP_FRAME_COUNT, resize, VideoWriter
from concurrent.futures import ThreadPoolExecutor
from os import path, makedirs
from mne import Annotations, read_evokeds
from mne.io.edf.edf import RawEDF
from utils.edf import EdfUtil
from shutil import rmtree
from ESA.A3D_model import R3DClassifier
from ESA.offline_process import get_stft_feature, plot_feature, plot_feature_map
from ESA.plot_result import plot_seid_res, plot_esa_res
from SeiD.two_feature_model import Classifier1D_2D_3D
from AD.gogogo import Art_Dec
from AD.APSD import APSD
from AD import setdata
import SD.Config as cfg
from SD.finnal3 import get_label_data
from SD.Premodel import getlabel
from SD.edf2mat import edf2mat, filter_2sIIR, anno_txt
from SD.mat2npz import mat2npz
from HFO.hfo import hfo_process
from VD.api import Colors, annotating_box, actionRecognition, non_max_suppression, scale_coords
from VD.config import Config
from VD.Pre_videodata import LoadVideos
from VD.Load_model import loadModel, DetectModule, ResNet


class StatisticsThread(QThread):
    """
    统计特征线程
    """
    res_signal = pyqtSignal(int, str, list)
    finish_signal = pyqtSignal()

    def __init__(self, g_or_r, group_idx, ch_idx, raw, freq=None, start_time=None, stop_time=None):
        super(StatisticsThread, self).__init__()
        self.g_or_r = g_or_r
        self.group_idx = group_idx
        self.ch_idx = ch_idx
        self.raw = raw
        self.freq = freq
        self.start_time = start_time
        self.stop_time = stop_time

    def calculate_statistics(self):
        """
        计算统计特征
        """
        if self.g_or_r == 'r':
            span_start_index = int(self.start_time * self.freq)
            ms = int(1 / self.freq * 1000)
            span_end_index = int(self.stop_time * self.freq) - ms
            raw_data = self.raw.get_data(start=span_start_index, stop=span_end_index)
            statistics_1 = self.calculate_from_raw(raw_data, self.ch_idx)
            spectrum = self.raw.compute_psd(tmin=self.start_time, tmax=self.stop_time)
            statistics_2 = EdfUtil.calculate_single_channel_avg_psd(spectrum, self.ch_idx)
            self.res_signal.emit(self.group_idx, 'r', [statistics_1, statistics_2])

        if self.g_or_r == 'g':
            statistics_1 = self.calculate_from_raw(self.raw.get_data(), self.ch_idx)
            spectrum = self.raw.compute_psd()
            statistics_2 = EdfUtil.calculate_single_channel_avg_psd(spectrum, self.ch_idx)
            self.res_signal.emit(self.group_idx, 'g', [statistics_1, statistics_2])

    @staticmethod
    def calculate_from_raw(raw_data, index):
        """
        计算指定通道mean、std、var
        """
        res = []
        ch_data = raw_data[index]
        ch_data_uv = ch_data * 10 ** 6
        res.append("%.2f" % mean(ch_data_uv))  # mean
        res.append("%.2f" % std(ch_data_uv))  # std
        res.append("%.2f" % var(ch_data_uv))  # var
        return res

    def run(self):
        self.calculate_statistics()
        self.finish_signal.emit()


def crop_tactic(raw, start_time, dur):
    """
    Presuming the largest sfreq reaches 1000Hz, crop the raw with margin (+-4s).
    (Notch Filter length: 6601 samples (6.601 s))

    AD/SD requires to add resultant annotations to the raw, so this tactic seems not advisable.
    """
    margin = 4

    t_max = raw.times[-1]
    sfreq = raw.info['sfreq']

    stop_time = start_time + dur - 1 / sfreq
    if start_time - margin < 0:
        crop_start = 0
    else:
        crop_start = start_time - margin
    if stop_time + margin > t_max:
        crop_stop = t_max
    else:
        crop_stop = stop_time + margin
    raw.crop(tmin=crop_start, tmax=crop_stop)
    return raw, start_time - crop_start


class SeiDESAThread(QThread):
    """
    SeiD/ESA 线程
    """
    fm_signal = pyqtSignal(bytes)
    feature_signal = pyqtSignal(bytes)
    res_signal = pyqtSignal(bytes)
    finish_signal = pyqtSignal()

    def __init__(self, raw, start_time, mod):
        super(SeiDESAThread, self).__init__()
        self.raw, self.pre_margin = crop_tactic(raw, start_time, 4)
        self.mod = mod

    def seid(self):
        model = Classifier1D_2D_3D(2, (2, 2, 2, 2), pretrained=True)
        checkpoint = load(AddressConfig.get_seid_adr('cp'), map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # 预处理
        raw = self.raw.notch_filter(freqs=50)
        raw = raw.filter(l_freq=1, h_freq=70)
        # print(raw.times[-1])

        # 切片2
        raw = raw.crop(self.pre_margin, self.pre_margin + 4 - 1 / raw.info['sfreq'])
        # print(pre_margin)
        # print(pre_margin + 4 - 1 / raw.info['sfreq'])

        # tmp3
        tmp3, power, data = get_stft_feature(raw, AddressConfig.get_seid_adr('STFT'))

        # tmp1
        tmp1 = data.transpose((1, 0))
        tmp1 = tensor(tmp1.astype('float32'))
        tmp1 = tmp1.reshape(1, 4000, 21)

        out1, out2, feature_map = model(tmp1, tmp3)

        # tmp1 = torch.randn(8, 4000, 21)
        # tmp3 = torch.randn(8, 3, 21, 32, 32)

        # feature map
        layer_label = ['input', 'conv1', 'block1', 'block2', 'block3', 'block4']
        feature_map.insert(0, tmp3)
        plot_feature_map(layer_label, feature_map, self.fm_signal)
        # (1, 3, 21, 32, 32)
        # (1, 16, 21, 16, 16)
        # (1, 16, 21, 9, 9)
        # (1, 32, 11, 5, 5)
        # (1, 64, 6, 3, 3)
        # (1, 128, 3, 2, 2)

        # (21, 32, 32, 3)
        # (21, 16, 16, 16)
        # (21, 9, 9, 16)
        # (11, 5, 5, 32)
        # (6, 3, 3, 64)
        # (3, 2, 2, 128)

        # feature
        plot_feature(power[2], self.feature_signal)
        # print('resnet:', out1.shape)
        # print('resnet:', out2.shape)
        # result
        plot_seid_res(out1, out2, self.res_signal)

        # p = sum(map(lambda p: p.numel(), model.parameters()))
        # print('parameters size:', p)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # net = model.to(device)
        # summary(net, input_size=[(4000, 21),(3, 21, 256,256),(3, 256, 256)])

    def esa(self):
        model = R3DClassifier(8, (2, 2, 2, 2), pretrained=True)
        checkpoint = load(AddressConfig.get_esa_adr('cp'), map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # 预处理
        raw = self.raw.notch_filter(freqs=50)
        raw = raw.filter(l_freq=1, h_freq=70)

        # 切片2
        raw = raw.crop(self.pre_margin, self.pre_margin + 4 - 1 / raw.info['sfreq'])

        stft_buffer, power, _ = get_stft_feature(raw, AddressConfig.get_esa_adr('STFT'), save=False)
        out, feature_map = model(stft_buffer)

        # feature map
        layer_label = ['input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        feature_map.insert(0, stft_buffer)
        plot_feature_map(layer_label, feature_map, self.fm_signal)

        # feature
        plot_feature(power[2], self.feature_signal)  # 只取索引为2用于绘图

        # result
        plot_esa_res(out, self.res_signal)

    def run(self):
        if self.mod == 'DSMN-ESS':
            self.seid()
        elif self.mod == 'R3DClassifier':
            self.esa()
        self.finish_signal.emit()


class ADThread(QThread):
    """
    AD 线程
    """
    topo_signal = pyqtSignal(bytes)
    res_signal = pyqtSignal(bytes)
    ann_signal = pyqtSignal(list, list, int, int)
    finish_signal = pyqtSignal()

    def __init__(self, raw, start_time, stop_time, mod1, mod2, arti_list, fb_idx, auto=False):
        super(ADThread, self).__init__()
        self.raw = raw
        self.start_time = start_time
        self.stop_time = stop_time
        self.mod1 = mod1
        self.mod2 = mod2
        self.arti_list = arti_list
        self.fb_idx = fb_idx
        self.auto = auto
        self.annotations_list = None
        self.des_list = None

    def ad(self):
        # 滤波
        raw1 = self.raw.notch_filter(freqs=50)
        raw_filtered = raw1.filter(l_freq=1, h_freq=70)

        # channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2']
        selected_indices = [0, 1, 10, 11, 12, 13, 14, 15, 8, 9]

        s_freq = int(raw_filtered.info['sfreq'])
        time = int(raw_filtered.n_times / s_freq)  # 1200 int 向下取整

        raw_data = raw_filtered.get_data()  # (19,1200000) 1200s
        # re_data = raw_data.reshape(time, 19, s_freq)  # (1200,19,1000) (总时长,通道数,采样频率)

        three_dim_array = []
        for i in range(time):
            start_index = i * 1000  # 计算起始索引
            end_index = (i + 1) * 1000  # 计算结束索引
            data_to_add = raw_data[:, start_index:end_index]
            three_dim_array.append(data_to_add)

        re_data = array(three_dim_array)

        Hdl_Var = re_data[self.start_time:self.stop_time]  # (11,19,1000)
        s_data = (Hdl_Var[:, selected_indices, :] * 1000000)  # (11,10,1000)
        X = setdata.dataset(s_data)  # (11,10,1000)

        n_data = from_numpy(raw_data)

        APSD(n_data, self.start_time, self.stop_time, self.fb_idx, self.topo_signal)  # toposize
        self.annotations_list, self.des_list = Art_Dec(X, self.arti_list, raw_filtered, self.start_time, self.mod1,
                                                       self.mod2, self.res_signal, self.auto)

    def run(self):
        self.ad()
        # 是否自动添加事件
        # 注：秉持最大限度保持edf内容的原则，只会将事件添加到edf，而接口内的预处理不会对edf造成影响，edf原本包含的事件也不会丢失
        if self.auto:
            self.ann_signal.emit(self.annotations_list, self.des_list, self.start_time, self.stop_time)
        self.finish_signal.emit()


class SDBaseThread(QThread):
    """
    SD 线程基类
    """
    swi_signal = pyqtSignal(str)
    res_signal = pyqtSignal(bytes, RawEDF)
    ann_signal = pyqtSignal(list, list, int, int)
    finish_signal = pyqtSignal()

    def __init__(self, raw, start_time, stop_time, auto=False):
        super(SDBaseThread, self).__init__()
        self.raw = raw.set_eeg_reference(ref_channels='average')  # 平均导联
        self.start_time = start_time
        self.stop_time = stop_time
        self.duration = self.stop_time - self.start_time
        self.auto = auto
        self.annotations_list = []

    def sd(self):
        pass

    def plot_eeg_and_save(self, raw, dur_th, scaling):
        text_size = 16
        font_family = "Microsoft YaHei"
        if raw.annotations:
            if self.duration >= dur_th:
                eeg_plot = raw.plot(duration=self.duration, show=False, scalings=scaling, start=self.start_time,
                                    show_scrollbars=False)
                raw.annotations.rename(mapping={'': 'SSW'})
                # Add SSW
                fig = eeg_plot.get_figure()
                fig.suptitle('SSW', fontsize=text_size, fontweight='bold', color='#1f77b4', y=0.98)
            else:
                raw.annotations.rename(mapping={'': 'SSW'})
                eeg_plot = raw.plot(duration=self.duration, show=False, scalings=scaling, start=self.start_time,
                                    show_scrollbars=False)
            # Add legend
            event_label = "SSW: Spike-slow wave"
            # 获取了绘图的第一个轴（Axes）对象并设置图例
            eeg_plot.get_axes()[0].legend([event_label], loc='lower center', bbox_to_anchor=(0.44, -0.18),
                                          frameon=False, prop={'family': font_family, 'size': text_size})
        else:
            eeg_plot = raw.plot(duration=self.duration, show=False, scalings=scaling, start=self.start_time,
                                show_scrollbars=False)

        # 获取第一个轴对象，通常包含图表的标题和注释
        ax = eeg_plot.mne.ax_main
        # 设置横轴label字体
        ax.set_xlabel(ax.get_xlabel(), fontproperties=font_family, fontsize=text_size)
        # 统一设置坐标轴刻度字体大小和颜色
        ax.tick_params(axis='both', which='major', labelsize=text_size, labelcolor=ThemeColorConfig.get_txt_color())
        # 遍历这个轴上的所有文本对象，设置字体
        for text in ax.texts:
            text.set_weight('bold')
            text.set_fontsize(text_size)

        # Save the figure
        buffer = self._save_fig_2_buffer(eeg_plot)

        # Crop the raw for showing in HoverLabel
        raw.crop(tmin=self.start_time, tmax=self.stop_time - 1 / 500)

        return buffer

    @staticmethod
    def _save_fig_2_buffer(fig):
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=600)
        buffer.seek(0)
        return buffer

    def run(self):
        self.sd()
        if self.auto:
            if len(self.annotations_list) > 0:
                self.ann_signal.emit(self.annotations_list, ['SSW'], self.start_time, self.stop_time)
        self.finish_signal.emit()


class SDTemplateThread(SDBaseThread):
    """
    SD 模板匹配线程
    """

    def __init__(self, raw, start_time, stop_time, auto=False):
        super(SDTemplateThread, self).__init__(raw, start_time, stop_time, auto)

    def sd(self):
        # 滤波
        raw1 = self.raw.notch_filter(freqs=50)
        raw_filtered = raw1.filter(l_freq=1, h_freq=70)

        # 清除annotations
        raw_filtered.set_annotations(Annotations([], [], []))

        # 加载模板数据
        template = read_evokeds(cfg.tempdata, verbose='error')[0]
        template_data = template.data  # (19,150)

        # 设置相关度阈值
        corr_threshold = 0.985

        # 初始化保存相关窗口的列表
        corr_windows = []

        # calculate SWI
        total_dur = 0

        # 滑动窗口匹配，窗口大小为 0.3 秒，步长为 0.01 秒
        for i in range(int(self.start_time) * 500, int(self.stop_time) * 500 - 150, 5):
            # 获取当前窗口内的数据
            window_data = raw_filtered.get_data(start=i, stop=i + 150)  # (19,150)
            # 计算当前窗口内的相关度
            corr = corrcoef(window_data, template_data)[0, 1]  # (19,150) 取[0,1]作为参考，不取[0,0]是因为对角线（变量自身与自身）的相关系数都是1
            abs_corr = abs(corr)  # 正相关和负相关都可以
            # 判断相关度是否超过阈值，如果超过则保存当前窗口的起始和终止时间
            if abs_corr > corr_threshold:
                corr_windows.append((i / 500, (i + 150) / 500))

        if len(corr_windows) > 0:
            for start, end in get_label_data(corr_windows):
                dur = end - start
                raw_filtered.annotations.append(start, dur, '')
                total_dur += dur

                if self.auto:
                    self.annotations_list.append([start, end - start, 'SSW'])

        self.swi_signal.emit('%.2f' % (total_dur / self.duration))

        buffer = self.plot_eeg_and_save(raw_filtered, 15, 300e-6)

        self.res_signal.emit(buffer.getvalue(), raw_filtered)


class SDSemanticsThread(SDBaseThread):
    """
    SD 语义分割线程
    """

    def __init__(self, raw, start_time, stop_time, auto=False):
        super(SDSemanticsThread, self).__init__(raw, start_time, stop_time, auto)

    def sd(self):
        # todo 50 Notch
        raw1 = self.raw.notch_filter(freqs=50)

        # 用于绘制
        raw_filtered = raw1.copy().filter(l_freq=1, h_freq=70)  # filter直接修改了原始对象，需要先copy

        # 切片
        sfreq = raw1.info['sfreq']
        raw1 = raw1.crop(self.start_time, self.stop_time - 1 / sfreq)
        record_microvolts = raw1.get_data() * 1e6  # 单位转换

        # 0.5-50 BP
        samplerate = 500  # in Hz
        passband = [0.5, 50]  # passband for bandpass filter
        forder = 6  # filter order
        record1 = filter_2sIIR(record_microvolts, passband, samplerate, forder, 'bandpass')

        # folder process
        mat_path = AddressConfig.get_sd_adr('mat')
        npz_path = AddressConfig.get_sd_adr('npz')
        self.clean_folder_list([mat_path, npz_path])
        # 如果这个目录或其任何父目录不存在的话，创建一个名为 mat_path 的目录。
        # 如果这个目录已经存在，代码不会有任何错误，而是正常继续执行。这在确保创建目录时不因目录已存在而中断程序执行的情况下非常有用。
        makedirs(mat_path, exist_ok=True)
        makedirs(npz_path, exist_ok=True)

        # edf2mat
        edf2mat(record1, mat_path)

        # 生成伪标签以满足数据格式条件
        anno_txt(mat_path, record1.shape[1], sfreq)  # record1.shape[1] = raw.n_times

        # mat2npz
        mat2npz(mat_path, npz_path)

        # calculate SWI
        total_dur = 0

        raw_filtered.set_annotations(Annotations([], [], []))

        # 得到npz后
        res = getlabel()
        if len(res) > 0:
            for start, end in res:
                real_dur = (end - start) / sfreq
                real_start = start / sfreq + self.start_time
                raw_filtered.annotations.append(real_start, real_dur, '')
                total_dur += real_dur

                if self.auto:
                    self.annotations_list.append([real_start, real_dur, 'SSW'])

        self.swi_signal.emit('%.2f' % (total_dur / self.duration))

        buffer = self.plot_eeg_and_save(raw_filtered, 15, 300e-6)

        self.res_signal.emit(buffer.getvalue(), raw_filtered)

    @staticmethod
    def clean_folder_list(folder_path_list):
        for p in folder_path_list:
            if path.exists(p):
                rmtree(p)


class HFOThread(QThread):
    """
    HFO 线程
    """
    res_signal = pyqtSignal(RawEDF)
    finish_signal = pyqtSignal()

    def __init__(self, raw, ch_idx, start_time, stop_time):
        super(HFOThread, self).__init__()
        self.raw = raw
        self.ch_idx = ch_idx
        self.start_time = start_time
        self.stop_time = stop_time

    def hfo(self):
        self.res_signal.emit(hfo_process(self.raw, self.ch_idx, self.start_time, self.stop_time))

    def run(self):
        self.hfo()
        self.finish_signal.emit()


class VDThread(QThread):
    """
    VD 线程
    Be careful that input_list can not be None。
    """
    img_signal = pyqtSignal(ndarray)
    res_signal = pyqtSignal(str)
    percent_signal = pyqtSignal(int)
    normal_finish_signal = pyqtSignal()
    abnormal_finish_signal = pyqtSignal()

    def __init__(self):
        super(VDThread, self).__init__()
        self.input_list = None
        self.save = False
        self.output_adr = None
        # self.s_or_m = None  # 0: SISO | 1: MIMO
        self.cfg = Config()
        self.conf_thres = 0.25  # 置信度
        self.iou_thres = 0.45  # iou
        self.jump_out = True  # 跳出循环
        self.is_continue = True  # 继续/暂停
        self.percent_length = 10000  # 进度条
        self.detectModel = None
        self.actionModel = None
        self.vid_cap = None
        self.last_box = None
        self.device = device(self.cfg.device)
        self.color = Colors()
        self.features = []
        self.output_frames = []
        self.cnt = 0
        self.thread_pool = None

    def update_adr(self, input_list, output_adr):
        self.input_list = input_list
        self.output_adr = output_adr

    def update_mod(self, detect_mod, action_mod):
        self.detectModel = detect_mod
        self.actionModel = action_mod

    def update_progress(self):
        if self.vid_cap:
            total_frames = self.vid_cap.get(CAP_PROP_FRAME_COUNT)
            percent = int(self.cnt / total_frames * self.percent_length)
            self.percent_signal.emit(percent)

            if self.cnt >= total_frames:
                self.percent_signal.emit(0)

    def process_image(self, img, img0, video_path):
        self.last_box, xyxy = self.detect(img, img0, self.last_box)
        im = img0.copy()
        if self.last_box is None:  # 未检测到患者！
            self.img_signal.emit(im)
            self.output_frames.append(im)
        else:
            out = annotating_box(img0, xyxy, color=self.color(1, True), line_width=self.cfg.line_thickness)
            self.img_signal.emit(out)
            self.output_frames.append(out)
            # 将处理后的图像中特定区域的特征保存起来
            self.features.append(resize(im[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], (112, 112)))

        if self.cnt % 60 == 0 and self.cnt > 0:  # 每三秒输出一个动作标签
            res = actionRecognition(self.actionModel, self.features, self.device)
            cur_second = self.cnt / 20
            self.res_signal.emit(f'{cur_second - 3}-{cur_second}s: {res} —> {path.basename(video_path)}')
            self.features = []

    def detect(self, img, img0, last_box):
        pred, featuremap = self.detectModel(img, augment=False, visualize=False)
        pred = non_max_suppression([pred[0], pred[2]], self.conf_thres, self.iou_thres, None, False, max_det=5)

        if pred is not None:
            pred = sorted(pred, key=lambda x: x[0][4], reverse=True)[0]
            last_box = pred.clone()
        elif last_box is not None:
            pred = last_box.clone()
        else:
            return None, None

        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
        *xyxy, conf, cls = pred[0]  # 星号表达式 将 pred[0] 中的前几个元素赋值给一个名为 xyxy 的列表
        return last_box, xyxy

    def video_changed(self, video_path, video_size):
        """
        callback func
        """
        self.last_box = None  # 清除上一帧检测出来的框
        self.features = []
        self.cnt = 0

        if self.save:
            if self.output_frames:
                # 决定使用的视频尺寸：如果 video_size 有效，则使用它；否则，从当前帧获取尺寸
                output_size = video_size if video_size else (self.output_frames[0].shape[1],
                                                             self.output_frames[0].shape[0])
                # if not self.s_or_m:
                #     output_adr = self.output_adr
                # else:
                output_name = path.splitext(path.basename(video_path))[0]  # 分割为基本名和扩展名
                output_adr = path.join(self.output_adr, output_name + '_VD_processed.mp4')

                output_frames = self.output_frames.copy()
                self.thread_pool.submit(self.run_vd_writer, output_adr, output_frames, output_size)

        self.output_frames = []

    @staticmethod
    def run_vd_writer(output_adr, output_frames, output_size):
        current_video_writer = VDWriteThread(output_adr, output_frames, output_size)
        current_video_writer.start()
        current_video_writer.wait()  # 阻塞当前线程，等待当前视频保存完成

    def run(self):
        # try:
        if len(self.input_list) > 5:
            workers = 5
        else:
            workers = len(self.input_list)
        self.thread_pool = ThreadPoolExecutor(max_workers=workers)

        stride = self.detectModel.stride
        dataset = LoadVideos(Config(), self.input_list, stride=stride, next_video_callback=self.video_changed)
        dataset = iter(dataset)
        while True:
            if self.jump_out:
                self.stop()
                self.abnormal_finish_signal.emit()
                break
            if self.is_continue:
                try:
                    p, img, img0, self.vid_cap = next(dataset)
                    self.cnt += 1  # 帧数
                except StopIteration:
                    self.stop()
                    self.normal_finish_signal.emit()
                    break
                self.update_progress()
                self.process_image(img, img0, p)  # dataset.count

        # except Exception as e:
        #     print(e)

    def stop(self):
        if self.vid_cap:
            self.vid_cap.release()
        self.features = []
        self.output_frames = []
        self.cnt = 0
        self.last_box = None
        self.percent_signal.emit(0)
        self.thread_pool.shutdown(wait=False)


class VDModelThread(QThread):
    """
    VD 模型线程
    """
    mod_signal = pyqtSignal(DetectModule, ResNet)
    finish_signal = pyqtSignal()

    def __init__(self, cfg):
        super(VDModelThread, self).__init__()
        self.cfg = cfg

    def run(self):
        detect_mod, action_mod = loadModel(AddressConfig.get_vd_adr('cp1'), AddressConfig.get_vd_adr('cp2'), self.cfg)
        self.mod_signal.emit(detect_mod, action_mod)
        self.finish_signal.emit()


class VDWriteThread(QThread):
    """
    VD 保存线程
    Be careful that output_size can not be None.
    """

    def __init__(self, output_adr, output_frames, output_size):
        super(VDWriteThread, self).__init__()
        self.output_adr = output_adr
        self.output_frames = output_frames
        self.output_size = output_size
        self.running = False

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        fourcc = VideoWriter.fourcc(*"mp4v")
        output_video = VideoWriter(self.output_adr, fourcc, 20, self.output_size)
        for frame in self.output_frames:
            if not self.running:
                break
            output_video.write(frame)

        output_video.release()
