from typing import List
from mne import create_info, io, channels
from mne.io.edf.edf import RawEDF
from mne.time_frequency import Spectrum
from mne.viz import plot_montage
from utils.config import ChannelEnum, MontageEnum, PSDEnum
from numpy import array, log10, mean, var, std, sqrt, min as np_min, max as np_max, nan
from matplotlib.pyplot import show, figure, plot, Figure
from antropy import sample_entropy, higuchi_fd, app_entropy
from concurrent.futures import ThreadPoolExecutor


class EdfUtil:
    """
    Edf工具类
    """

    @staticmethod
    def get_montage():
        position = {}
        for i in range(19):
            ch_name = ChannelEnum.TPM.value[i]
            pos = [MontageEnum.XPOS.value[i], MontageEnum.YPOS.value[i], MontageEnum.ZPOS.value[i]]
            position[ch_name] = array(pos)
        montage = channels.make_dig_montage(ch_pos=position)
        return montage

    @classmethod
    def show_montage(cls):
        plot_montage(cls.get_montage())
        show()

    @staticmethod
    def build_collected_21CH_mapping():
        ori_order = ChannelEnum.COLLECTED.value
        tar_order = ChannelEnum.CH21.value
        mapping = [tar_order.index(ch) for ch in ori_order]
        return mapping

    @staticmethod
    def build_raw(data, freq):
        info = create_info(
            ch_names=ChannelEnum.CH21.value,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', ],
            sfreq=freq)
        custom_raw = io.RawArray(data, info)
        return custom_raw

    @staticmethod
    def map_channels(selected_channels: List, raw_channels: List):
        """
        将raw中的通道映射为统一的通道名
        :param selected_channels: 用户选择的目标通道
        :param raw_channels: EDF文件中实际的通道
        :return: 更新后的 selected_channels, 匹配的 mapping_list
        """
        # 获得selected_channels在raw_edf上的映射：mapping_list（目标：数量相等，名字对应）
        tmp_channels = selected_channels.copy()
        mapping_list = []
        for key in tmp_channels:
            tmp = []
            for ch_name in raw_channels:
                if key in ch_name:
                    tmp.append(ch_name)
            if len(tmp) == 0:  # 选择 > 存在
                selected_channels.remove(key)
            elif len(tmp) == 1:  # 选择 = 存在
                mapping_list.extend(tmp)
            else:  # 选择 < 存在：存在相近的通道名，则只保留最短的
                tmp = [ch_name for ch_name in tmp if len(ch_name) == min(len(ch) for ch in tmp)]
                if len(tmp) == 1:
                    mapping_list.extend(tmp)
                else:  # 存在两条名字相同的通道：直接报异常
                    mapping_list = []
                    break
        return selected_channels, mapping_list

    @classmethod
    def normalize_edf(cls, edf_path: str, selected_channels: List, diary=None):
        """
        按规定的通道名及通道顺序调整edf格式
        """
        raw = io.read_raw_edf(edf_path, preload=True)
        raw_channels = raw.info['ch_names']
        if diary:
            diary.info(f'EDF_adr: {edf_path}\n'
                       f'EDF_chns: {raw_channels}')

        selected_channels, mapping_list = cls.map_channels(selected_channels, raw_channels)
        if len(mapping_list) != len(selected_channels):
            error_msg = (
                f"Error: Length mismatch between mapping_list ({len(mapping_list)}) "
                f"and selected_channels ({len(selected_channels)}). "
                f"Mapping channels: {mapping_list}, Selected channels: {selected_channels}"
            )
            if diary:
                diary.error(error_msg)
            raise ValueError(error_msg)

        # 读取映射表内的通道
        # raw = io.read_raw_edf(edf_path, preload=True, include=mapping_list)
        raw.reorder_channels(mapping_list)  # 按映射表调整通道顺序，去除多余的通道
        # raw.pick(mapping_list)  # Pick a subset of channels.

        # 修改通道名
        dic = {}
        for i in range(len(mapping_list)):
            dic[mapping_list[i]] = selected_channels[i]
        raw.rename_channels(dic)
        # raw.pick(picks=selected_channels)  # 会改变raw的通道，且通道名称会按list的指定顺序排列
        return raw

    @staticmethod
    def plot_eeg(raw: RawEDF) -> Figure:
        """
        绘制raw在0-30s内的所有通道
        """
        n_channels_to_plot = len(raw.ch_names)  # 绘制全部通道
        start, stop = raw.time_as_index([0, min(30, raw.times[-1])])  # 选取 0~30 秒的数据
        times = raw.times[start:stop]  # 获取时间轴
        # times = raw.times  # 获取完整的时间点
        fig = figure(figsize=(4, 3))  # 创建图表
        ax = fig.add_subplot(111)

        # 遍历选定的通道进行绘制
        for i in range(n_channels_to_plot):
            channel_data, _ = raw[i, start:stop]  # 获取通道数据
            # channel_data, _ = raw[i, :]  # 获取通道数据
            plot(times, channel_data[0] * 1e6, label=raw.ch_names[i])  # 直接绘制曲线

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        # ax.legend(loc="upper right", fontsize="small", ncol=2)  # 添加通道图例
        fig.tight_layout()
        return fig

    @staticmethod
    def normal_filter(raw: RawEDF):
        """
        50Hz、1-70Hz
        """
        return raw.notch_filter(freqs=50).filter(l_freq=1, h_freq=70)

    @classmethod
    def calculate_all_channels_avg_psd(cls, spectrum: Spectrum, dB=False) -> List[int]:
        """
        计算所有通道平均功率谱密度
        整个二维数组所有数值加起来，除以元素总数，全局平均
        粒度：所有通道 × 所有频段 → 全局平均
        就是对calculate_band_all_channels_avg_psd按频段循环后再求一次通道上的平均
        """
        avg_psd = []
        for fmin, fmax in PSDEnum.FREQ_BANDS.value:
            # avg_psd_uv2 = spectrum.get_data(fmin=freq_band[0], fmax=freq_band[1]).mean() * (10 ** 6) ** 2
            avg_psd_uv2 = cls.calculate_band_all_channels_avg_psd(spectrum, fmin, fmax).mean()
            avg_psd.append(10 * log10(avg_psd_uv2) if dB else avg_psd_uv2)
        return avg_psd

    @classmethod
    def calculate_single_channel_avg_psd(cls, spectrum: Spectrum, index: int) -> List[str]:
        """
        计算指定通道平均功率谱密度（表示在每个频带中电压波动的平方均值，分布在每赫兹的频率上。）
        [index] 提取了某一个通道的数据，所以此时的数组是一维的，不需要指定 axis
        粒度：单通道，所有频段
        就是对calculate_band_all_channels_avg_psd按频段循环后取某个通道的索引值
        """
        avg_psd = []
        for fmin, fmax in PSDEnum.FREQ_BANDS.value:
            avg_psd_uv2 = cls.calculate_band_all_channels_avg_psd(spectrum, fmin, fmax)[index]
            avg_psd.append("{:.2e}".format(avg_psd_uv2))  # 科学计数法（指数表示法）
        return avg_psd

    @staticmethod
    def calculate_band_all_channels_avg_psd(spectrum: Spectrum, fmin, fmax) -> List[int]:
        """
        计算所有通道在指定频段内的平均功率谱密度
        粒度：所有通道，单频段
        """
        avg_psd_uv2 = spectrum.get_data(fmin=fmin, fmax=fmax).mean(axis=1) * (10 ** 6) ** 2
        return avg_psd_uv2  # shape: (n_channels, n_freqs) (19,1025) -> (n_channels,)

    @staticmethod
    def calculate_mse(raw: RawEDF, max_scale=20, chns=None) -> List[float]:
        """
        并行计算 MSE（多尺度熵），支持单个通道或多个通道平均
        MSE 需要先对信号进行不同尺度的降采样（Coarse-graining），然后在每个尺度上计算 sample_entropy()（单尺度样本熵）
        """
        data = raw.get_data(picks=chns).mean(axis=0) * 1e6  # 所有通道取均值并转换为 µV

        def calculate_sampen(scale):
            """
            计算样本熵（SampEn）
            """
            coarse_grained = mean(data[:len(data) // scale * scale].reshape(-1, scale), axis=1)  # 粗粒化处理
            return sample_entropy(coarse_grained, order=2)

        with ThreadPoolExecutor() as executor:
            # range(1, max_scale + 1) 产生 [1, 2, ..., max_scale]
            # 然后 executor.map(compute_sampen, ...) 会并行传递 scale 给 compute_sampen(scale) 进行计算
            # executor.map() 按照 range(1, max_scale + 1) 的顺序调用 calculate_sampen(scale)。
            # 无论线程执行快慢，最终 mse_values[i] 对应 scale=i+1，顺序不变！
            # 等价于 mse_values = [compute_sampen(scale) for scale in range(1, max_scale + 1)]
            # 区别是：使用 ThreadPoolExecutor.map() 会并行执行 compute_sampen(scale)，而列表推导式是顺序执行
            mse_values = list(executor.map(calculate_sampen, range(1, max_scale + 1)))

        return mse_values

    @classmethod
    def calculate_brain_region_mse(cls, raw: RawEDF, region_channels: dict[str, List[str]]) -> dict[str, List[float]]:
        """
        计算多个脑区的 MSE
        """
        mse_res = {}
        for region_name, chns in region_channels.items():
            mse_res[region_name] = cls.calculate_mse(raw, chns=chns)
        return mse_res

    @staticmethod
    def calculate_features(raw: RawEDF, chn_name: str) -> List[float]:
        """
        并行计算单通道的多个统计特征
        """
        data = raw.get_data(picks=chn_name)[0] * 1e6  # 转换为 µV

        def calculate_feature(func, *args):
            """
            通用特征计算
            """
            try:
                return func(*args)
            except Exception as e:
                print(f"特征计算错误: {func.__name__} -> {e}")
                return nan

        # 特征计算任务
        tasks = [
            (np_min, data),  # 最小值
            (np_max, data),  # 峰值
            (var, data),  # 方差
            (std, data),  # 标准差
            (lambda d: sqrt(mean(d ** 2)), data),  # 均方根（RMS）
            (lambda d: np_max(d) / sqrt(mean(d ** 2)) if sqrt(mean(d ** 2)) != 0 else nan, data),  # 裕度因子 = 峰值 / RMS
            (app_entropy, data),  # 近似熵（ApEn）
            (sample_entropy, data),  # 样本熵（SampEn）
            (higuchi_fd, data)  # 分形维数（Higuchi Fractal Dimension）
        ]

        with ThreadPoolExecutor() as executor:
            res = list(executor.map(lambda t: calculate_feature(t[0], t[1]), tasks))

        return res
