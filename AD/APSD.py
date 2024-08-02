from numpy import arange
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mne import create_info, EvokedArray
from matplotlib.pyplot import savefig, close
from io import BytesIO
from utils.config import ChannelConfig


def APSD(data, tmin, tmax, fb_idx, topo_signal=None):  # size
    channel_config = ChannelConfig()

    # 创建info对象
    info = create_info(ch_names=channel_config.topomap_channels, sfreq=1000., ch_types='eeg')
    # print(info)

    # 创建evokeds对象
    evoked = EvokedArray(data, info)
    if fb_idx != 0:
        freq_band = channel_config.freq_bands[fb_idx - 1]
        evoked.filter(freq_band[0], freq_band[1])

    # evokeds设置通道
    if tmax is None:
        times = tmin
    else:
        times = arange(tmin, tmax)  # size
    evoked.set_montage(channel_config.get_montage())
    tpm = evoked.plot_topomap(times, ch_type="eeg", show=False, nrows=3, ncols=4)

    buffer = BytesIO()
    savefig(buffer, format='png', dpi=300)
    close()
    buffer.seek(0)
    if topo_signal is not None:
        topo_signal.emit(buffer.getvalue())
    else:
        return buffer.getvalue()
