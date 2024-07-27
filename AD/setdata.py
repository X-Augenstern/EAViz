from torch import from_numpy, stack
from AD.config import config


def dataset(data):
    # 对data做z-score标准化后转换为Tensor并还原回原始维度
    s_data_list = []
    for t in range(data.shape[0]):
        slice_data = data[t]  # 提取时间点 t 的二维子数组  (10,1000)
        processed_data = norm(slice_data)  # 调用 norm 函数处理二维子数组
        s_data_list.append(from_numpy(processed_data))  # 将处理后的数据添加到列表中 Tensor
    s_data = stack(s_data_list)  # 在第一维度对list内的张量进行堆叠 二—>三维 (11,10,1000)
    return s_data


def norm(slice_data):
    # z-score标准化
    mean = config.mean
    std = config.std
    for i in range(slice_data.shape[0]):
        slice_data[i] = (slice_data[i] - mean[i]) / std[i]
    return slice_data
