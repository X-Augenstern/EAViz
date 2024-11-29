from mne import create_info, io, channels
from mne.viz import plot_montage
from utils.config import ChannelEnum, MontageEnum
from numpy import array
from matplotlib.pyplot import show


def get_montage():
    position = {}
    for i in range(19):
        ch_name = ChannelEnum.TPM.value[i]
        pos = [MontageEnum.XPOS.value[i], MontageEnum.YPOS.value[i], MontageEnum.ZPOS.value[i]]
        position[ch_name] = array(pos)
    montage = channels.make_dig_montage(ch_pos=position)
    return montage


def show_montage():
    plot_montage(get_montage())
    show()


def build_collected_21CH_mapping():
    ori_order = ChannelEnum.COLLECTED.value
    tar_order = ChannelEnum.CH21.value
    mapping = [tar_order.index(ch) for ch in ori_order]
    return mapping


def build_raw(data, freq):
    # todo ESA通道顺序调整（完成）
    info = create_info(
        ch_names=ChannelEnum.CH21.value,
        ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                  'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', ],
        sfreq=freq)
    custom_raw = io.RawArray(data, info)
    return custom_raw
