from numpy import arange
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mne import create_info, EvokedArray
from matplotlib.pyplot import savefig, close, gcf
from io import BytesIO
from utils.config import PSDEnum, ChannelEnum
from utils.edf import EdfUtil


def APSD(data, tmin, tmax, fb_idx, topo_signal=None):  # size
    # 创建info对象
    info = create_info(ch_names=ChannelEnum.TPM.value, sfreq=1000., ch_types='eeg')
    # print(info)

    # 创建evokeds对象
    evoked = EvokedArray(data, info)
    if fb_idx != 0:
        freq_band = PSDEnum.FREQ_BANDS.value[fb_idx - 1]
        evoked.filter(freq_band[0], freq_band[1])

    # evokeds设置通道
    if tmax is None:
        times = tmin
    else:
        times = arange(tmin, tmax)  # size
    evoked.set_montage(EdfUtil.get_montage())

    text_size = 16

    tpm = evoked.plot_topomap(times, ch_type="eeg", show=False, nrows=3, ncols=4)
    fig = tpm if hasattr(tpm, 'axes') else gcf()

    # 放大子图标题及 color_bar 刻度
    for ax in fig.axes:
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=text_size)

    # 放大 colorbar
    cbar_ax = fig.axes[-1]  # 通常最后一个axes是colorbar
    cbar_ax.tick_params(labelsize=text_size)

    buffer = BytesIO()
    savefig(buffer, format='png', dpi=300)
    close()
    buffer.seek(0)
    if topo_signal is not None:
        topo_signal.emit(buffer.getvalue())
    else:
        return buffer.getvalue()
