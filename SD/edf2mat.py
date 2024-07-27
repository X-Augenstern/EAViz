from numpy import zeros_like
from random import uniform
from os import path, makedirs
from scipy.io import savemat
from scipy.signal import butter, filtfilt


# raw = mne.io.read_raw_edf('liang_19_filtered.edf', preload=True)  # 单位为伏特，而matlab中的edfread单位为微伏，转换出来的npz文件也为微伏
# record = raw.load_data()

# edfbroswer软件的50Hz陷波滤波功能暂时不能替代，需要导入预先处理过的filtered数据
# raw0 = mne.io.read_raw_edf('liang_19.edf', preload=True)
# record0 = raw0.load_data()
# raw0_filter = raw0.copy()
# raw0_filter.load_data()
# raw0_filter.notch_filter(freqs=50)
# a=record0.get_data()
# b=record.get_data()
# c=raw0_filter.get_data()

# 单位转换
# record_microvolts = record.get_data() * 1e6


# --------------------------------------------------------------------------#
# Matlab滤波过程
def filter_2sIIR(sig, f, fs, n, type='low'):
    if filter_2sIIR.__code__.co_argcount < 4:  # 获取了 filter_2sIIR 函数的参数数量。这是Python的内省特性之一，可以用于获取函数的元信息。
        raise ValueError('Not enough input arguments! please check')
    # elif filter_2sIIR.__code__.co_argcount < 5:
    #     type = 'low'

    # if float(f) >= float(fs)/2:
    #     raise ValueError(
    #         'The sampling frequency is not adequate for the given cutoff frequency. Please input a lower f.')

    if type.lower() == 'bandpass':
        # Highpass
        [b, a] = butter(n, f[0] / (fs / 2), 'high')
        sigfilter = zeros_like(sig)
        for i in range(sig.shape[0]):
            sigfilter[i, :] = filtfilt(b, a, sig[i, :], padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))

        # Lowpass
        [b, a] = butter(n, f[-1] / (fs / 2), 'low')
        for i in range(sigfilter.shape[0]):
            sigfilter[i, :] = filtfilt(b, a, sigfilter[i, :], padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    else:
        [b, a] = butter(n, f / (fs / 2), type)
        sigfilter = zeros_like(sig)
        for i in range(sig.shape[0]):
            sigfilter[i, :] = filtfilt(b, a, sig[i, :])

    return sigfilter


# # 文件路径和保存路径
#
# mat_path = "matdata"
# # 创建保存npz文件的文件夹（如果不存在）
# os.makedirs(mat_path, exist_ok=True)
# file_name = "liang_19_filtered.edf"
#
# # 初始化
# samplerate = 500  # in Hz
# passband = [0.5, 50]  # passband for bandpass filter
# forder = 6
# # 进行滤波处理
# record1 = filter_2sIIR(record_microvolts, passband, samplerate, forder, 'bandpass')
#
# # 如果数据需要转置则取消注释
# # record=-record
#
def edf2mat(filtered_data, mat_path):
    save_file_path = path.join(mat_path, 'temp' + ".mat")
    savemat(save_file_path, {"data": filtered_data})


# ---------------------------生成伪标签以满足数据格式条件--------------------------------------------
def anno_txt(mat_path, raw_n_times, raw_freq):
    # 指定.txt文件的保存路径
    txt_path = mat_path
    # 创建保存txt文件的文件夹（如果不存在）
    makedirs(txt_path, exist_ok=True)
    txt_file_name = "temp_annotations.txt"
    # 拼接保存路径
    txt_file_path = path.join(txt_path, txt_file_name)
    total_times = raw_n_times / raw_freq
    # 指定注释数据
    annotations = """Onset,Duration,Annotation
    +0.0000000,{:.7f},NREM1\n""".format(total_times)

    # 生成Spike注释数据
    spike_annotations = []
    for _ in range(int(total_times / 2)):
        onset = round(uniform(0, total_times), 7)  # 生成一个在两个给定值之间的随机浮点数并四舍五入到小数点后七位
        duration = round(uniform(0, 1), 4)
        spike_annotations.append((onset, duration))
    # 对注释数据按照Onset进行排序
    spike_annotations.sort(key=lambda x: x[0])
    # 将注释数据拼接到annotations字符串中
    for onset, duration in spike_annotations:
        annotations += "+{:.7f},{:.4f},Spike\n".format(onset, duration)
    # 将注释数据写入txt文件
    with open(txt_file_path, "w") as f:
        f.write(annotations)

# #-----------------------直接将.mat文件存为.npz（npz数据相同，但是模型识别效果差），仍使用script函数------------------------#
# # 指定.mat文件的路径
# file_path = r'D:\资料\睡眠分期+棘波\实时系统\代码\matdata1\liang_19_filtered.mat'
#
# # 指定保存npz文件的路径
# npz_path = 'npzdata1'
#
# # 采样率和每个文件的时间长度
# sampling_rate = 500
# segment_length = 30  # 单位：秒
#
# # 读取.mat文件
# data = sio.loadmat(file_path)
#
# # 获取数据
# eeg_data = data['data']  # 假设脑电数据在.mat文件中的变量名为'eeg_data'
#
# # 计算总样本数和每个文件的样本数
# total_samples = eeg_data.shape[1]
# samples_per_segment = sampling_rate * segment_length
#
# # 确定要生成的文件数量
# num_files = total_samples // samples_per_segment
#
# # 创建保存npz文件的文件夹（如果不存在）
# os.makedirs(npz_path, exist_ok=True)
#
# # # 创建自定义的睡眠分期标签数据
# # stage_label_data = 1  # 自定义睡眠分期标签数据
# # # 创建自定义的棘波标签数据
# # spike_label_data = np.array([[ 6731,  6994],
# #  [ 7085,  7213],
# #  [ 8239,  8363],
# #  [ 9020,  9151],
# #  [10291, 10848],
# #  [11344, 11610],
# #  [11896, 12514],
# #  [12936, 13046],
# #  [13267, 13477],
# #  [14128, 14261],
# #  [14334, 14441],
# #  [14527, 14688]])  # 自定义棘波标签数据
#
# # 将数据分段保存为npz文件
# for i in range(num_files):
#     start_index = i * samples_per_segment
#     end_index = start_index + samples_per_segment
#     # segment_data = {
#     #     'data':eeg_data[:, start_index:end_index],
#     #     'stage_label':stage_label_data,
#     #     'spike_label': spike_label_data
#     # }
#     segment_data = eeg_data[:, start_index:end_index]
#     save_file_path = os.path.join(npz_path, f'data_segment_{i+1}.npz')
#     # np.savez(save_file_path, **segment_data)
#     np.savez(save_file_path, data=segment_data)
#
#     print(f'Saved segment {i+1}/{num_files} to {save_file_path}')
