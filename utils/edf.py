from typing import List
from mne import create_info, io, channels
from mne.viz import plot_montage
from utils.config import ChannelEnum, MontageEnum
from numpy import array
from matplotlib.pyplot import show


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
        # todo ESA通道顺序调整（完成）
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
    def normalize_edf(cls, edf_path: str, selected_channels: List, diary):
        """
        按规定的21通道名及通道顺序调整edf格式
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
