from os import listdir, path
from torch import load, from_numpy, no_grad, argmax
from SD.Unet34 import Unet34
from numpy import zeros, where, array, load as np_load
import SD.Config as cfg


def prepareModel(cfg):
    model = Unet34(n_channels=cfg.channel, SA=cfg.SA)  # 带SA channel=1 all in：19  注意SA
    state_dict = load(cfg.testweight)['state_dict']
    model.load_state_dict(state_dict)
    # print(f'loading model {cfg.model} with weight {cfg.testweight}')
    return model.float().cuda()


def pair2label(s_pairs, L, th):
    new_pairs = []
    for s_pair in s_pairs:
        if s_pair[1] - s_pair[0] >= th:  # 持续时间超过给定阈值
            new_pairs.append(s_pair)  # 从第一次的预测中取出符合要求的时间段并添加到新的集中

    label = zeros(L)  # (15000,)
    for s_pair in new_pairs:
        label[s_pair[0]:s_pair[1]] = 1  # 将符合要求的时间段值置为1
    return label


# 输出的是 class 'numpy.ndarray'  实际上和获得的标签只是转换尺寸而已 比如从 1*12 转5*2

def label2Spair(label):
    d_label = label[1:] - label[:-1]  # label去掉第一个元素  减去  label 去掉最后一个元素  输出是全零 shape (14999,0)
    # a = np.where(d_label != 0)  # 返回一个包含所有非零元素索引的ndarray的元组 即tuple(ndarray)
    inds = list(where(d_label != 0)[0] + 1)  # 输出包含了数组d_label中非零元素的秒数

    if label[0] == 1:
        inds.insert(0, 0)
    if label[-1] == 1:
        inds.append(len(label))

    s_pair = array(inds)
    s_pair = s_pair.reshape(-1, 2)  # (x,2)
    return s_pair


def getlabel():
    # edf -> mat -> npz 每个npz全长15000
    # 逐个遍历npz (19,15000) -> normalize -> unsqueeze (1,19,15000) -> <model> (1,2,15000) -> argmax (15000,)
    # -> 2pair (x,2) -> 2label (15000,) -> 2pair (y,2) -> + 15000*t (y,2) -> 2list[list] y[2] -> res -> t+1
    # 第二遍2pair应该就是为了得到(⭐,2)的形式，方便后续+ 15000*t

    model2 = prepareModel(cfg)  # 加载模型
    model2.eval()

    folder_path = cfg.npzdata  # 获取npz文件夹
    result = []
    t = 0
    for filename in listdir(folder_path):  # 遍历文件夹中的所有.npz文件
        npzpath = path.join(folder_path, filename)
        input2 = from_numpy(np_load(npzpath)['data']).cuda().float()  # 从npz读取key = data的value Tensor (19,15000)
        # print('实际标签')
        # print(np.load(npzpath)['spike_label'])

        for i in range(len(cfg.mean)):  # 对不同通道进行归一化
            input2[i] = (input2[i] - cfg.mean[i]) / cfg.std[i]
        input2 = input2.unsqueeze(0)  # (1,19,15000) 作为模型输入
        # batch_size = input2.shape[0]  # 1

        with no_grad():
            U34pred = model2(input2)['seg_out']  # (1,2,15000)
        U34pred = argmax(U34pred, dim=1).squeeze(0).cpu().numpy()  # (15000,)
        U3_s_pair = label2Spair(U34pred)  # (x,2)
        U3pred_new = pair2label(U3_s_pair, 15000, 15)  # (15000,)
        prelable = label2Spair(U3pred_new)  # (y,2)
        # print('预测标签')
        # print(prelable)
        prelable_modified = [[x + 15000 * t for x in sublist] for sublist in prelable]  # (y,2) -> list[list] y[2]
        result.extend(prelable_modified)  # 添加一个可迭代对象中的所有元素到列表末尾
        t += 1
    return result

# input2 = torch.from_numpy(np.load(r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪0607\liang01_19_labeled_filtered_00030.npz")['data']).cuda().float()  # 标准化后的19通道信号
# print('实际标签')
# print(np.load(r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪0607\liang01_19_labeled_filtered_00030.npz")['spike_label'])
#
# for i in range(len(cfg.mean)):  # 不同通道的归一化操作
#     input2[i] = (input2[i] - cfg.mean[i]) / cfg.std[i]
# input2 = input2.unsqueeze(0)
# #print(input2)
# batch_size = input2.shape[0]
# model2 = PrepareModel(cfg)
# model2.eval()
#
# # for 循环遍历 30s数据  19通道数据
# with torch.no_grad():
#     U34pred = model2(input2)['seg_out']  # # input = torch.randn([1,1, 15000])  # ([3, 1, 15000]) 修改过输出了
# U34pred = torch.argmax(U34pred, dim=1).squeeze(0).cpu().numpy()  # print(pred)
# U3_s_pair = label2Spair(U34pred)  # print(s_pair)
# U3pred_new = pair2label(U3_s_pair, 15000, 15)
# res = label2Spair(U3pred_new)
# print('预测标签')
# print(res)
