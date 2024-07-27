from io import BytesIO
from cv2 import imdecode, IMREAD_COLOR, resize
from mne import time_frequency
from numpy import asarray, float32, empty, uint8, squeeze, transpose, flipud
from seaborn import heatmap
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from torch import tensor


def plot_feature_map(layer_label, feature_map, fm_signal):
    """
    Plot the feature map results of each convolutional layer during the detection process to show.
    :param layer_label: list of each layer
    :param feature_map: list[list] of the result of each layer
    :param fm_signal: emit bytes of the plotted feature map
    """
    plt.figure(figsize=(12, 8))
    for i in range(len(layer_label)):
        ax = plt.subplot(2, 3, i + 1)
        im = feature_map[i]
        # [N, C, D, H, W] -> [C, D, H, W] 将batch维度压缩掉，detach阻断反向梯度传播
        # Batch C（RGB通道） D（EEG通道） H（时间） W（频率）
        im = squeeze(im.detach().numpy(), axis=0)
        # [C, D, H, W] -> [D, H, W, C] 改变各维度
        im = transpose(im, [1, 2, 3, 0])
        # (21, 32, 32, 3)
        # (21, 16, 16, 64)
        # (11, 9, 9, 64)
        # (6, 5, 5, 128)
        # (3, 3, 3, 256)
        # (2, 2, 2, 512)
        plt.imshow(flipud(im[1, :, :, 1]), origin='lower')  # 垂直翻转图像数据再与y轴一起翻转
        plt.title(f'{layer_label[i]}', fontsize=16)
        # y轴已在get_stft_feature中经过了翻转，此处不需要再翻转

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    fm_signal.emit(buffer.getvalue())
    plt.close()


def plot_feature(power_slice, feature_signal):
    """
    Plot one STFT figure to show.
    :param power_slice: power in one channel
    :param feature_signal: emit bytes of the plotted feature figure
    """
    plt.figure()
    sns_plot = heatmap(power_slice, cmap='jet', cbar=False)
    sns_plot.invert_yaxis()  # 反转Y轴坐标，即将最高值置于顶部，最低值置于底部
    # 获取当前的 x 轴刻度位置
    # xtick_positions = sns_plot.get_xticks()
    # xtick_labels = [f'{i / 5}' for i in range(len(xtick_positions))]  # 将刻度位置除以5
    # sns_plot.set_xticklabels(xtick_labels)
    # 定义想要显示的刻度位置 减半 [0.5, 4.5, 8.5, 12.5, 16.5, 20.5, 24.5, 28.5, 32.5, 36.5]
    selected_xtick_positions = sns_plot.get_xticks()[::2]
    # 设置新的 x 轴刻度位置
    sns_plot.set_xticks(selected_xtick_positions)
    # 生成与选定刻度位置对应的标签
    selected_xtick_labels = [f'{i / 5:.1f}' for i in range(0, 20, 2)]
    # 设置新的 x 轴刻度标签
    sns_plot.set_xticklabels(selected_xtick_labels)

    # 设置新的 y 轴刻度位置和标签
    ytick_positions = sns_plot.get_yticks()
    ytick_labels = [f'{i * 5}' for i in range(len(ytick_positions))]  # 将刻度位置乘以5
    sns_plot.set_yticklabels(ytick_labels)

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # 获取热力图对象的所有子对象，然后找到与热力图相关的映射对象
    mappable = sns_plot.get_children()[0]
    cbar = plt.colorbar(mappable, label='Magnitude')
    cbar.outline.set_visible(False)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    feature_signal.emit(buffer.getvalue())
    plt.close()


def get_stft_feature(raw, adr_stft, save=False):
    """
    Get the STFT feature of raw.
    :param raw: time span == 4s with 1-70, 50Hz filtering
    :param adr_stft: save path of the generated STFT figures
    :param save: save or not
    :return: stft_buffer, power, data
    """
    # data, times = raw[:]  # data：(21,4000) times：(4000)
    data = raw.get_data()
    data = data * 10 ** 6  # mat读取edf的单位为微伏，mne读取edf的单位为V

    # 短时傅里叶变换（STFT）算法是一种将信号在时间和频率上进行局部分析的算法。
    # 它将信号分成多个小段，并对每个小段进行傅立叶变换，从而得到该时间段内信号的频谱信息。这样可以获得信号在不同时间和频率上的特征
    # 1、分帧：首先，将待分析的信号分成多个长度相等的窗口，通常使用汉明窗（Hamming Window）来减小边界效应。
    # 2、加窗：对每个窗口内的数据进行加窗操作，这可以确保时域信号在频域分析时的平滑过渡，减少泄漏效应。
    # 3、傅立叶变换：对加窗后的每个窗口内的数据进行傅立叶变换，得到频域上的信号表示。
    # 4、重叠与合并：为了获得更好的时间分辨率，通常将相邻窗口的数据进行一定程度的重叠和合并操作，以平滑信号过渡。
    power = time_frequency.stft(data, 200)  # 200：Length of the STFT window in samples (must be a multiple of 4).
    power = abs(power)  # (21,101,40)
    power = power[:, 0:16, :]  # (21,16,40) (通道，频率点，时间点)

    # fs = 1000
    # nperseg = 200
    # noverlap = 95
    # frequencies, times, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Zxx = abs(Zxx)
    # Zxx = Zxx[:, 0:16, :]  # (21,16,40)

    # 保存通道索引为3的STFT图
    # plt.figure()
    # C 参数的维度与 X 和 Y 参数的维度不匹配。C 是颜色值或数据，X 和 Y 是用于定义网格的坐标值。在STFT的情况下，X 代表时间轴，Y 代表频率轴，而 C 代表STFT的幅度或强度。
    # plt.pcolormesh(times, frequencies[0:16], Zxx[2], shading='auto', cmap='jet')
    # plt.title('STFT Magnitude')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Frequency [Hz]')
    # plt.colorbar(label='Magnitude')
    # plt.savefig('./STFT_3.png')

    # A3D-EEG_epoch-1.pth.tar  (64,1,3,7,7) - input: input[1, 1, 21, 32, 32]
    # A3D-EEG_epoch-19.pth.tar (64,3,3,7,7) - input: input[1, 3, 21, 32, 32]
    # 预处理
    guiyihua = transforms.ToTensor()
    suofang = transforms.Resize([32, 32])
    # zuhe
    zuhe_transform = transforms.Compose([  # 定义一个预处理方法组合
        transforms.ToPILImage(),  # 将样本数据张量转换为PIL图像，以便使用PIL库进行图像处理
        suofang,  # 按照比例把图像最小的一个边长放缩到32，另一边按照相同比例放缩。
        guiyihua,  # 转换为tensor格式，这个格式可以直接输入进神经网络了
    ])

    stft_buffer = empty((21, 3, 32, 32), float32)
    for i in range(21):
        plt.figure(figsize=(256 / 80, 256 / 80))
        # plt.pcolormesh(times, frequencies[0:16], Zxx[i], shading='auto', cmap='jet')
        sns_plot = heatmap(power[i], cmap='jet', cbar=False)
        sns_plot.invert_yaxis()  # 反转Y轴坐标，即将最高值置于顶部，最低值置于底部
        buffer = BytesIO()
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format='png', dpi=80)
        if save:
            plt.savefig(adr_stft + f'STFT_{i}.png', dpi=80)
        buffer.seek(0)
        # 从内存缓冲区读取数据 —> 将数据转换为字节数组 —> 转换为 NumPy 数组 —> 从这些字节数据中解码出图像
        im_data = imdecode(asarray(bytearray(buffer.read()), dtype=uint8), IMREAD_COLOR)  # (256,256,3)
        im_data_resized = resize(im_data, (112, 112))  # (112,112,3)
        plt.close()
        im_data = zuhe_transform(im_data_resized)  # (3,32,32)
        im_data = im_data.numpy()  # (3,32,32)
        stft_buffer[i] = im_data
    stft_buffer = stft_buffer.transpose((1, 0, 2, 3))
    stft_buffer = tensor(stft_buffer.astype('float32')).reshape(1, 3, 21, 32, 32)
    # 1
    # img = img.reshape(1, 1, 21, 32, 32)
    # 19
    # img = img.reshape(1, 3, 21, 32, 32)
    return stft_buffer, power, data

    # 1
    # total_img = np.empty((21, 32, 32))
    # for i in range(21):
    #     img = power[i, 0:16, :].astype('uint8')
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     img = zuhe_transform(img)
    #     img = img.numpy()[0, :, :]
    #     total_img[i, :, :] = img
    # return total_img

    # 19
    # total_img = np.empty((21, 3, 32, 32))  # (21,32,32)
    # for i in range(21):
    #     img = power[i, 0:16, :]  # (16,40)
    #     img = img.astype('uint8')
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # (16,40,3)
    #     img = zuhe_transform(img)  # (3,32,32)
    #     # img = img.numpy()[0, :, :]  # (32,32)
    #     total_img[i, :, :, :] = img.numpy()  # (21,32,32)
    # total_img = total_img.transpose((1, 0, 2, 3))
    # return total_img
