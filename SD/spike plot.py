import numpy as np

# from SWI_09 import temmplate_matching
from SWI_20 import slowWaveDetector

import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


# 设置西文字体为新罗马字体
from matplotlib import rcParams

config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False #解决负号无法显示的问题
}
rcParams.update(config)

from model.Unet34 import Unet34


# 主要是进行多模型的绘图操作

def iou(anchor, pred):
    if pred[1] <= anchor[0] or pred[0] >= anchor[1]:
        iou = 0  # 预测末端小于真值首值 或 预测首值大于真值末端
    else:
        left = min(anchor[0], pred[0])
        right = max(anchor[1], pred[1])
        in_left = max(anchor[0], pred[0])
        in_right = min(anchor[1], pred[1])
        iou = (in_right - in_left) / (right - left)
    return iou


def get_intersection(label_s, label_e, pred_s, pred_e):
    if pred_s < label_s:
        pred_s = label_s
    if pred_e > label_e:
        pred_e = label_e
    return pred_e - pred_s


def match_pairs_segment(label, pred, iou_th):
    pred_pairs = label2Spair(pred)  # 标签格式变更
    label_pairs = label2Spair(label)  # 标签格式变更 M*N  M 棘波个数 N start 和 end
    print('预测标签')
    print(pred_pairs/15000*30)
    print('真实标签')
    print(label_pairs/15000*30)
    TP = 0
    FP = 0
    match_dict = dict()
    for label in label_pairs:
        label_s, label_e = label
        match_dict[label_s] = 0
        for pred in pred_pairs:
            pred_s, pred_e = pred
            if pred_s > label_e:
                break
            if label_e >= pred_s >= label_s or label_e >= pred_e >= label_s or (label_e <= pred_e and label_s >= pred_s):
                match_dict[label_s] += get_intersection(label_s, label_e, pred_s, pred_e)
        if match_dict[label_s] / (label_e - label_s) >= iou_th:
            TP += 1
        else:
            FP += 1

    return label_pairs.shape[0], pred_pairs.shape[0], TP, FP  # num_T 真实的棘波数, num_P 预测棘波数, num_TP , num_FP


def match_pairs(label, pred, iou_th):
    pred_pairs = label2Spair(pred)  # 标签格式变更
    label_pairs = label2Spair(label)  # 标签格式变更 M*N  M 棘波个数 N start 和 end
    tp_T = 0
    for label_pair in label_pairs:
        for pred_pair in pred_pairs:
            iou_pair = iou(label_pair, pred_pair)
            if iou_pair >= iou_th:
                tp_T = tp_T + 1  # 若存在 标签和预测的匹配值是大于阈值的，tp_T 加一

    tp_P = 0
    # print(len(pred_pairs))
    # print(pred_pairs)
    # print(len(label_pairs))
    # print(label_pairs)
    for pred_pair in pred_pairs:
        for i in range(len(label_pairs)):
            iou_pair = iou(label_pairs[i], pred_pair)
            if iou_pair >= iou_th:
                break
            if i == len(label_pairs) - 1:
                tp_P = tp_P + 1  #

    return label_pairs.shape[0], pred_pairs.shape[0], tp_T, tp_P  # num_T 真实的棘波数, num_P 预测棘波数, num_TP , num_FP


def evalu(label, pred, iou_th, sample="Unkown"):
    num_T, num_P, num_TP, num_FP = match_pairs_segment(label, pred, iou_th)
    # 真实棘波数  预测棘波数  真棘波预测为棘波数  正常样本被预测为棘波数
    # if num_TP > 50 or num_P > 50:
    print('棘慢波标签数', num_T)
    print('预测的棘波标签数', num_P)
    print('预测正确的棘慢波数', num_TP)
    print('预测错误的棘慢波数', num_FP)
    #     print(sample)
    #     exit(0)
    Sens = num_TP / (num_T + 10e-6)  # Sens = Recall = TP/TP+FN
    Prec = num_TP / (num_P + 10e-6)  # Prec = TP/TP+FP
    # Fp_min = (num_P - num_FP) / (len(pred) / 500 / 60)
    # Fp_min = (num_FP) / (len(pred) / 500 / 60)
    # Fp_min = num_FP*2
    return Sens, Prec, num_FP

    # 抹除太短的111标签，返回一个 1*15000的值


def pair2label(s_pairs, L, th):
    new_pairs = []
    for s_pair in s_pairs:
        if s_pair[1] - s_pair[0] >= th:  # 长度超过最低值
            new_pairs.append(s_pair)  # 从第一次的预测中取出长度符合要求的值并添加到新的集中

    label = np.zeros(L)
    for s_pair in new_pairs:
        label[s_pair[0]:s_pair[1]] = 1
    return label


# 输出的是 class 'numpy.ndarray'  实际上和获得的标签只是转换尺寸而已 比如从 1*12 转5*2

def label2Spair(label, start=0):
    d_label = label[1:] - label[:-1]  # label去掉第一个元素  减去  label 去掉最后一个元素  输出是全零 shape (14999,0)
    inds = list(np.where(d_label != 0)[0] + 1)

    if label[0] == 1:
        inds.insert(0, 0)
    if label[-1] == 1:
        inds.append(len(label))

    s_pair = np.array(inds) + start
    s_pair = s_pair.reshape(-1, 2)
    return s_pair


# ----------------------------输出部分-------------------------------------------------
def plot_label_pred(signal, spike, sqnn_and_mf_pred, sqnn_segment, events, U3pred_new):
    # 原始信号、棘波标签、融合输出、单独SQNN输出、形态学输出、U3P

    # ------------------------------------输出结果可视化彩色曲线 5channel  -----------------------------------
    X = np.linspace(0, 30, 15000)
    x = X
    y = signal  # 标准化后的信号
    # ----------上色用的数组-------------------------------------------------------------
    colors_label = []       # 真实标签
    colors_Unet = []        # 单独U-NET输出
    colors_morphology = []  # 单独形态学 12
    colors_Template = []    # 单独模板匹配 09
    colors_Unet_and_MF = [] # 融合 U-NET+ 形态学
    colors_U3pluspred = []  # U-NET3+
    # ------------------标签------------------
    for cou1 in range(0, 15000, 1):
        if spike[cou1] == 1:  # Spike为标签
            colors_label.append('#FF0000')
        else:
            colors_label.append('#87CEFA')  # 底色蓝色

    # --------------单独 U-NET---------------
    for cou2 in range(0, 15000, 1):
        if sqnn_segment[cou2] == 1:
            colors_Unet.append('#008000')
        else:
            colors_Unet.append('#87CEFA')

    # --------------单独形态学----------------
    for cou3 in range(0, 15000, 1):
        if events[cou3] == 1:
            colors_morphology.append('#FF0000')
        else:
            colors_morphology.append('#87CEFA')
    # -------------单独模板匹配-----------------
    # for cou4 in range(0, 15000, 1):
    #     if templatepred[cou4] == 1:
    #         colors_Template.append('#008000')
    #     else:
    #         colors_Template.append('#87CEFA')

    # --------------融合 U-NET 形态学---------
    for cou5 in range(0, 15000, 1):
        if sqnn_and_mf_pred[cou5] == 1:  # 两个模型的输出直接进行叠加  可考虑在 SQNN的基础上验证是否存在，然后进行叠加
            colors_Unet_and_MF.append('#008000')
        else:
            colors_Unet_and_MF.append('#87CEFA')

    # -------------- U-NET 3+---------------- 名称未改 是multitask
    for cou6 in range(0, 15000, 1):
        if U3pred_new[cou6] == 1:
            colors_U3pluspred.append('#FF0000')
        else:
            colors_U3pluspred.append('#87CEFA')
    # -------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    ax_label = plt.subplot(511)  # 标签
    ax_SQNN_pred = plt.subplot(512)  # U-NET
    ax_morphology = plt.subplot(513)  # 形态学
    # ax_Unet_and_MF_pred = plt.subplot(514)  # U-NET+形态学
    ax_Template = plt.subplot(514)  # 模板匹配
    ax_U3pluspred = plt.subplot(515)  # U-NET 3Plus  Unet34 修改接口即可

    # ------------------标签------------------
    points_label = np.array([X, y]).T.reshape(-1, 1, 2)
    segments_label = np.concatenate([points_label[:-1], points_label[1:]], axis=1)  # 数组拼接
    lc_label = LineCollection(segments_label, linewidths=1.5, color=colors_label)
    ax_label.set_title('Label')
    ax_label.set_xlim(min(x), max(x))
    ax_label.set_ylim(min(y), max(y))
    ax_label.set_yticks([])
    ax_label.add_collection(lc_label)

    # --------------单独 U-NET---------------
    points_spikepred = np.array([X, y]).T.reshape(-1, 1, 2)
    segments_spikepred = np.concatenate([points_spikepred[:-1], points_spikepred[1:]], axis=1)  # 数组拼接
    lc_spikepred = LineCollection(segments_spikepred, linewidths=1.5, color=colors_Unet)
    ax_SQNN_pred.set_title('SQNN')
    ax_SQNN_pred.set_xlim(min(x), max(x))
    ax_SQNN_pred.set_ylim(min(y), max(y))
    ax_SQNN_pred.set_yticks([])
    ax_SQNN_pred.add_collection(lc_spikepred)

    # --------------单独 形态学---------------
    points_morphologypred = np.array([X, y]).T.reshape(-1, 1, 2)
    morphology_spikepred = np.concatenate([points_morphologypred[:-1], points_morphologypred[1:]], axis=1)  # 数组拼接
    lc_morphologspikepred = LineCollection(morphology_spikepred, linewidths=1.5, color=colors_morphology)
    ax_morphology.set_title('Morphology')
    ax_morphology.set_xlim(min(x), max(x))
    ax_morphology.set_ylim(min(y), max(y))
    ax_morphology.set_yticks([])
    ax_morphology.add_collection(lc_morphologspikepred)

    # --------------模板匹配09---------
    # points_Template = np.array([X, y]).T.reshape(-1, 1, 2)
    # Template_pred = np.concatenate([points_Template[:-1], points_Template[1:]], axis=1)  # 数组拼接
    # lc_pred = LineCollection(Template_pred, linewidths=1.5, color=colors_Template)
    # ax_Template.set_title('Template matching')
    # ax_Template.set_xlim(min(x), max(x))
    # ax_Template.set_ylim(min(y), max(y))
    # ax_Template.set_yticks([])
    # ax_Template.add_collection(lc_pred)

    # # --------------融合 U-NET 形态学---------
    # points_pred = np.array([X, y]).T.reshape(-1, 1, 2)
    # segments_pred = np.concatenate([points_pred[:-1], points_pred[1:]], axis=1)  # 数组拼接
    # lc_pred = LineCollection(segments_pred, linewidths=1.5, color=colors_Unet_and_MF)
    # ax_Unet_and_MF_pred.set_title('SQNN+Morphology')
    # ax_Unet_and_MF_pred.set_xlim(min(x), max(x))
    # ax_Unet_and_MF_pred.set_ylim(min(y), max(y))
    # ax_Unet_and_MF_pred.set_yticks([])
    # ax_Unet_and_MF_pred.add_collection(lc_pred)

    # --------------Multi-task--------------------
    points_U3Plus_pred = np.array([X, y]).T.reshape(-1, 1, 2)
    U3Plus_pred = np.concatenate([points_U3Plus_pred[:-1], points_U3Plus_pred[1:]], axis=1)  # 数组拼接
    lc_U3Plusspikepred = LineCollection(U3Plus_pred, linewidths=1.5, color=colors_U3pluspred)
    ax_U3pluspred.set_title('Proposed Method')
    ax_U3pluspred.set_xlim(min(x), max(x))
    ax_U3pluspred.set_ylim(min(y), max(y))
    ax_U3pluspred.set_yticks([])
    ax_U3pluspred.add_collection(lc_U3Plusspikepred)

    plt.show()  # 最终绘图输出


if __name__ == "__main__":
    # 可以进行单个输出显示 联合 SQNN 和 形态学---------------------------
    import numpy as np
    from tools import PrepareModel, Transform, PrepareModel2
    import Config.Config as cfg
    from test import evaluate

    np.set_printoptions(threshold=np.inf)

    import math

    #  初始化----------------------------
    # 默认选取单通道的信号是 T3 channel
    mean = -0.0015852695649372956
    std = 31.562467510679348
    npz_path0 = r"D:\资料\睡眠分期+棘波\SWINET\sleep_discharge_dataset_9_19_channel\谌嘉诚_00140.npz"
    npz_path1 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\李亚楠0919\li_19_labeled_filtered_00034.npz"
    npz_path2 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪50\liang(1)_19_labeled_filtered_00140.npz"
    npz_path3 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪0607\liang_19_labeled_filtered_00017.npz"
    npz_path02 = r"D:\资料\睡眠分期+棘波\实时系统\代码\npzdata1\data_segment_7.npz"  # matlab程序转python测试使用
    npz_path03 = r"D:\资料\睡眠分期+棘波\实时系统\代码\npzdatatest\liang_19_filtered_00001.npz"   #matlab程序转python测试使用
    npz_path04 = r"D:\资料\睡眠分期+棘波\实时系统\代码\npzdata\liang_19_labeled_filtered_00001.npz" #matlab程序转python测试使用
    npz_path4 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪0607\liang01_19_labeled_filtered_00030.npz"
    npz_path5 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪0607\liang02_19_labeled_filtered_00064.npz"
    npz_path6 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\鲁婉宁0620\lu_19_labeled_filtered_00022.npz"
    npz_path7 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\莫其烨0830\mo_19_labeled_filtered_00051.npz"
    npz_path8 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\莫其烨0830\mo_invert_19_labeled_filtered_00051.npz"
    npz_path9 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\梁圣豪50\liang(1)_invert_19_labeled_filtered_00140.npz"
    #无标签数据
    npz_path10 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\卢怡朵0913\luyiduo_19_filtered_00042.npz"
    npz_path11 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\江19.08.09\jiang_invert_19_filtered_00051.npz"
    npz_path12 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\孙欣怡0824\sun_19_filtered_00031.npz"
    npz_path13 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\李21.03.18\limuhan01_19_filtered_00118.npz"
    npz_path14 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\李21.03.26\limuhan02_19_filtered_00017.npz"
    npz_path15 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\江19.08.18\jiang1_invert_19_filtered_00033.npz"
    npz_path16 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\莫其烨0106\mo1_invert_19_filtered_00031.npz"
    npz_path17 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\孙欣怡0810\sun1_invert_19_filtered_00041.npz"
    npz_path18 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\梁21.11.11\liang(2)_invert_19_filtered_00012.npz"
    npz_path19 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\李21.09.29\limuhan03_19_filtered_00041.npz"
    npz_path20 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\李21.11.19\limuhan04_19_filtered_00072.npz"
    npz_path21 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\李22.04.27\limuhan05_19_filtered_00066.npz"
    npz_path22 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\李22.05.11\limuhan06_19_filtered_00042.npz"
    npz_path23 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\孙欣怡20.08.19\sun2_19_filtered_00088.npz"
    npz_path24 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\汪19.11.14\wang01_19_filtered_00056.npz"
    npz_path25 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\汪20.01.15\wang02_19_filtered_00082.npz"
    npz_path26 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\汪20.04.21\wang03_19_filtered_00061.npz"
    npz_path27 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\汪21.03.03\wang04_19_filtered_00066.npz"
    npz_path28 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\杨19.08.05\yang01_19_filtered_00042.npz"
    npz_path29 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\杨19.10.14\yang02_19_filtered_00042.npz"
    npz_path30 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\郑20.12.06\zheng01_19_filtered_00042.npz"
    npz_path31 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\用药\郑20.12.14\zheng02_19_filtered_00042.npz"
    npz_path301 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\工具箱测试\zheng01_19_filtered_00042.npz"#工具箱测试使用
    npz_path32 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\朱凯东19.06.09\zhu01_19_filtered_00059.npz"
    npz_path33 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\朱凯东19.09.14\zhu02_19_filtered_00092.npz"
    npz_path34 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\朱凯东19.12.20\zhu03_19_filtered_00042.npz"
    npz_path35 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\程雨绮19.08.03\cheng_19_filtered_00066.npz"
    npz_path36 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\方铭泽19.07.08\fang_19_filtered_00077.npz"
    npz_path37 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\黄祺翔19.08.25\huang_19_filtered_00088.npz"
    npz_path34 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\朱凯东19.12.20\zhu03_19_filtered_00042.npz"
    npz_path34 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\朱凯东19.12.20\zhu03_19_filtered_00042.npz"
    npz_path34 = r"D:\资料\睡眠分期+棘波\裁剪数据\npz数据\滤波后\朱凯东19.12.20\zhu03_19_filtered_00042.npz"

    # 选择信号-----------------------------------------
    pathofnpz = npz_path37
    data = np.load(pathofnpz, allow_pickle=True)  # 首先导入 19 ch 的信号 根据模型的输入不同，进行数据的选择
    # print(data.files)  # 单个npz文件中 包含 数据 睡眠标签 棘波标签  ['data', 'stage_label', 'spike_label']
    # ----------------------不同模型的数据产生--------------------------
    p_ch_dict = {'谌嘉诚': 12, '方铭泽': 13, '孙弘浩': 13, '徐子宜': 4, '叶梦琪': 7, '易可欣': 12, '陈羽文': 12, '陈修泽': 12, '吴配瑶': 12}
    signal = data['data'] if data['data'].shape == (15000,) else data['data'][12]  # 取单通道数据 单通道模型使用、绘图使用 12 T3
    # signal = signal*(-1)
    data1 = signal  # 给形态学使用 独立读取信号
    Tdata = signal
    data19 = data  # 给19 通道模型使用 net以及Unet34
    # --------------统一获得各个标签--------------------------------
    stage_label = data['stage_label']  # 0 1 2
    spike_list = data['spike_label']  # 每一个棘波的位置点组合  [m,2]
    spike_label = np.zeros(15000)  # 产生0空矩阵，后续list转换后进行填充
    # -------------棘波标签list转01向量-----------------------------
    for listofspike in spike_list:
        start = listofspike[0]
        end = listofspike[1]
        spike_label[start:end + 1] = 1

    # ----------------单通道信号进行归一化处理SQNN-------------------------
    signal = (signal - mean) / std  # 使用深度网络才需要使用  原始信号进行归一化处理
    ss = signal  # 标准化后的信号  standard sigenal
    signal = torch.from_numpy(signal).unsqueeze(0)  # 函数转换为张量形式
    signal = signal.unsqueeze(0)  # 函数转换为张量形式   # print(signal.shape)  1 1 15000
    # ----------------多通道信号进行归一化处理Unet34-------------------------
    X = data19['data']
    Mean = cfg.mean
    Std = cfg.std
    for i in range(len(Mean)):  # 不同通道的归一化操作
        X[i] = (X[i] - Mean[i]) / Std[i]
    Y = X  # <class 'numpy.ndarray'>
    X = torch.from_numpy(X).unsqueeze(0)  # 函数转换为张量形式  <class 'torch.Tensor'> 1 19 15000

    # ------------单个测试-----------
    from tools import PrepareModel, Transform, PrepareModel2
    # import Config.Config as cfg
    from test import evaluate

    # --------第一模型载入 直接使用 preparemodel2 给SQNN用------------ Config设置 model2 = SpikeRegressor
    input1 = signal  # 标准化后的单通道信号
    input1 = input1.cuda().float()  # 数据集转入cuda # print(input.shape)
    batch_size = input1.shape[0]  # print(input)
    model = PrepareModel2(cfg, train=False)  # 给单纯的SQNN
    model.eval()
    # -----------------------------------------------------------
    # 第二模型载入 给Unet34 使用----------------------------   Config设置 model1=UNET34
    input2 = X # 标准化后的19通道信号
    input2 = input2.cuda().float()
    batch_size = input2.shape[0]
    model2 = PrepareModel(cfg, train=False)
    # model2 = Unet34(n_channels=cfg.channel, SA=cfg.SA).cuda().float()
    model2.eval()
    # ---------------------------------------------------
    # 获得SQNN模型输出--------------------------------------
    with torch.no_grad():
        SQNN_pred = model(input1)  # # input = torch.randn([1,1, 15000])  # ([3, 1, 15000])

    SQNN_pred = torch.argmax(SQNN_pred, dim=1).squeeze(0).cpu().numpy()  # print(pred)
    SQNN_s_pair = label2Spair(SQNN_pred)  # print(s_pair)

    # 获得 Unet34 标签---------------------------------------
    with torch.no_grad():
        U34pred = model2(input2)['seg_out']  # # input = torch.randn([1,1, 15000])  # ([3, 1, 15000]) 修改过输出了

    U34pred = torch.argmax(U34pred, dim=1).squeeze(0).cpu().numpy()  # print(pred)
    U3_s_pair = label2Spair(U34pred)  # print(s_pair)
    U3pred_new = pair2label(U3_s_pair, 15000, 15)
    # ----------------获得SQNN标签-------------
    SQNN_pred_new = pair2label(SQNN_s_pair, 15000,
                               10)  # 最后值是lmin 去除太短的  # print('\npred_new:') # print(type(pred_new)) # print(pred_new)
    SQNN_segment = SQNN_pred_new + 0  # 防止pred_net后续的变化影响segment 拷贝一份新的给segment
    # 标签格式变更 M*N  M 棘波个数 N start 和 end
    # ----------------获得形态学标签-------------
    events = slowWaveDetector(data1, spike_label)  # 原始标签
    events_copy = events
    events = pair2label(events[0], 15000, 10)  # 去掉过短的标签  有问题 采用合并代码再重新取得输出
    events_new = np.zeros(15000)  # 01 空矩阵用
    # for event, pred in zip(events, predlist):
    #     print(event)
    #     print(pred)
    # -----------------获得模板匹配标签-----------------------
    # templateevents = temmplate_matching(Tdata)
    # templatepred = np.zeros(15000)
    # for listofspike in templateevents:
    #     start = listofspike[0]
    #     end = listofspike[1]
    #     templatepred[start:end + 1] = 1
    # 合并慢波和棘波  + 附带获得events标签  ---------------------------------------
    for k, v in events_copy.items():
        v1 = pair2label(v, 15000, 10)  # 慢波部分转 0 1
        for cnt in range(len(v1)):
            if v1[cnt] == 1:
                # SQNN_pred_new[cnt] = 1  # 慢波的1 叠加到棘波上 SQNN  逻辑合并时注释掉
                events_new[cnt] = 1

    # 条件棘慢波合并  ---------------------------------------
    for k, v in events_copy.items():
        v1 = pair2label(v, 15000, 10)  # 慢波部分转 0 1
        for cnt2 in range(len(v1)):
            if v1[cnt2] == 1 and SQNN_pred_new[cnt2 - 5] == 1:
                SQNN_pred_new[cnt2] = 1  # 慢波的1 叠加到棘波上 SQNN

    # U-NET3+--------------------------------------------

    # 单个指标计算 SWI_error--------------------------------
    swi_label = np.mean(spike_label) * 100  # 标签值
    swi_SQNN = np.mean(SQNN_segment) * 100  # SQNN 预测值
    swi_M = np.mean(events_new) * 100  # 形态学
    swi_SQNN_and_M = np.mean(SQNN_pred_new) * 100  # SQNN+M 预测值  预测值
    swi_U3P = np.mean(U3pred_new) * 100  # U3P
    # -------------------------------------------------------

    SWI_error = abs(swi_U3P - swi_label)
    # Sens, Prec, Fp_min = evalu(spike_label, U3pred_new, 0.2)
    Sens, Prec, num_Fp = evalu(label=spike_label, pred=U3pred_new, iou_th=cfg.IoU)

    # print('SWI_true: ', swi_label)
    # print('SWI_pred: ', swi_U3P)
    print('SWI_error: ', SWI_error)
    print('Sens: ', Sens)
    print('Prec: ', Prec)
    print('Fp_min: ', num_Fp)

    plot_label_pred(ss, spike_label, SQNN_pred_new, SQNN_segment, events_new, U3pred_new)  # 可增加输入个数   1信号 2标签


