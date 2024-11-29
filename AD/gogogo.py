import torch
from torch.nn.functional import mse_loss
from numpy import interp, arange, zeros, concatenate, multiply
from AD.tools import PrepareModel1, PrepareModel2
from AD.config import config
from AD.AE_config import MyConfig  # 导入AE配置文件
from mne import Annotations
from mne.viz import plot_raw
from io import BytesIO
from utils.config import ThemeColorConfig

cfg = config()
mycfg = MyConfig()


def Art_Dec(Hdl_Var, arti_list, raw, st, mod1, mod2, res_signal, auto=False):
    cfg.model = mod1
    cfg.get_test_weight()

    mycfg.model = mod2
    mycfg.get_test_weight()

    model1 = PrepareModel1(cfg, test=True)  # 深度模型 DenseNet
    model2 = PrepareModel2(mycfg, mycfg.testweight)  # AE 模型 AutoEncoder

    model1.eval()
    model2.eval()

    Preds = []
    AE_index = []

    with torch.no_grad():
        outAE = 0
        X = Hdl_Var  # (11,10,1000)
        batch_size = len(X)  # 11
        X = X.float().cuda()

        # 深度模型
        pred = model1(X)  # (11,6)
        # 异常检测
        X = X.float().cuda()
        if mycfg.model != 'VAE':
            X1 = zeros((X.shape[0], X.shape[1], X.shape[2] + 24))  # (11,10,1024)
            for i in range(X.shape[0]):
                Y = []
                for j in range(X[i].shape[0]):
                    a = arange(0, 1000) + 1  # 1~1000 (1000,)
                    b = arange(0, 999, 1000 / 1025) + 1  # 1~1000 (1024,)
                    N = X[i, j, :].cpu().numpy()  # (1000,)
                    c = interp(b, a, N)  # 线性插值得到b在(a,N)上的y值 (1024,)
                    Y.append(c)
                X1[i, :, :] = Y
            X1 = torch.from_numpy(X1).float().cuda()  # (11,10,1024)
            outAE = model2(X1)  # (11,10,1024)
            # 求不同的Mse
            MSE = torch.mean(mse_loss(X1, outAE[0], reduction='none'), dim=2)  # (11,10)
        else:
            output = model2(X)
            for i in range(5):
                outAE += output[0][i]
            outAE = outAE / 5
            MSE = torch.mean(mse_loss(X, outAE, reduction='none'), dim=2)
        # 映射

        Area = torch.zeros(batch_size, 3)  # (11,3)
        Area[:, 0] = (MSE[:, 0] + MSE[:, 1]) / 2
        Area[:, 1] = (MSE[:, 2] + MSE[:, 3] + MSE[:, 4] + MSE[:, 5]) / 4
        Area[:, 2] = (MSE[:, 6] + MSE[:, 7] + MSE[:, 8] + MSE[:, 9]) / 4
        AE_preds = []
        for i in range(batch_size):
            torch.sigmoid(Area)
            aa = Area[i, :].tolist()  # 类型转换
            aa.append(0.95)  # 将阈值拼接到一起  阈值所在位置为 3
            Temp = torch.tensor(aa)
            index = Temp.sort(0, True).indices.numpy()  # 排序数据  并得到其下标
            # 阈值最大的情况
            if index[0] == 3:  # 正常数据   000001
                tempa = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float64)
            elif index[1] == 3:  # 阈值第二大的情况
                if index[0] == 0:  # 额区异常 110000
                    tempa = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float64)
                elif index[0] == 1:  # 颞区异常001100
                    tempa = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float64)
                else:  # 全局异常000010
                    tempa = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float64)
            elif index[2] == 3:  # 阈值第三大的情况
                if index[3] == 0:  # 额区颞区均异常111100
                    tempa = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float64)
                elif index[3] == 1:  # 颞区强异常001100
                    tempa = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float64)
                else:  # 全局异常000010
                    tempa = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float64)
            elif index[3] == 3:  # 阈值最小的情况 全局异常000010
                tempa = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float64)
            AE_preds.append(tempa.unsqueeze(0))
        # 合并每次数据
        AE_preds = torch.cat(AE_preds, 0)
        # Labels.append(label.cpu().numpy())  # 真实标签累计
        Preds.append(torch.sigmoid(pred).cpu().numpy())  # 深度模型预估标签累计 tensor to list
        AE_index.append(AE_preds.cpu().numpy())  # 异常检测模型预估标签累计

    dp_pred = concatenate(Preds, 0)  # 深度模型预测
    ae_pred = concatenate(AE_index, 0)  # Ae生成

    # y_true = np.concatenate(Labels, 0)  # 真实标签
    y_pred = multiply(ae_pred, dp_pred)

    y_pred = (y_pred > 0.1).astype(int)

    # 模型检测结果不变，只要遍历arti_list即可
    all_merged_annotations = []
    des_list = []
    for i, num in enumerate(arti_list):
        a_times = []
        description = None
        if num == 1:
            description = 'EB'
            des_list.append(description)
            for y in range(y_pred.shape[0]):
                data = tuple(y_pred[y])  # 获取当前时间点的数据

                if data[0] == 1 and data[5] == 0:
                    a_times.append(y)
            if a_times == []:
                print('无眨眼伪迹！')

        elif num == 2:
            description = 'FE'
            des_list.append(description)
            for y in range(y_pred.shape[0]):
                data = tuple(y_pred[y])  # 获取当前时间点的数据

                if data[1] == 1 and data[2] == 0 and data[5] == 0:
                    a_times.append(y)
            if a_times == []:
                print('无额区肌电！')

        elif num == 3:
            description = 'CE'
            des_list.append(description)
            for y in range(y_pred.shape[0]):
                data = tuple(y_pred[y])  # 获取当前时间点的数据

                if data[2] == 1:
                    a_times.append(y)
            if a_times == []:
                print('无咀嚼伪迹！')

        elif num == 4:
            description = 'TE'
            des_list.append(description)
            for y in range(y_pred.shape[0]):
                data = tuple(y_pred[y])  # 获取当前时间点的数据

                if data[3] == 1 and data[2] == 0 and data[5] == 0:
                    a_times.append(y)
            if a_times == []:
                print('无颞区肌电！')

        elif num == 5:
            description = 'Unclear'
            des_list.append(description)
            for y in range(y_pred.shape[0]):
                data = tuple(y_pred[y])  # 获取当前时间点的数据

                if data == (0, 0, 0, 0, 1, 0):
                    a_times.append(y)
            if a_times == []:
                print('无异常脑电！')

        elif num == 0:
            description = 'Normal'
            des_list.append(description)
            for y in range(y_pred.shape[0]):
                data = tuple(y_pred[y])  # 获取当前时间点的数据

                if data[5] == 1 and data[2] == 0:
                    a_times.append(y)
            if a_times == []:
                print('无正常脑电！')

        n_segments = []  # 不在连续部分中的值，dur为1s
        segments = []  # 存储连续部分的起始值和终止值 [5,6]，dur为end-start+1s
        start_time = None  # 连续部分的起始值 5

        for i in range(len(a_times) - 1):  # [1,3,5,6]
            if a_times[i + 1] - a_times[i] > 1:  # 每个检测到的时间点持续事件为1s，所以将时间间隔为1s的时间点视为连续部分
                if start_time is not None:
                    end_time = a_times[i]
                    segments.append((start_time, end_time))
                    start_time = None
            else:
                if start_time is None:
                    start_time = a_times[i]

        # 检查最后一个连续部分
        if start_time is not None:
            end_time = a_times[-1]
            segments.append((start_time, end_time))

        # 打印结果
        for segment in segments:
            start, end = segment
            print(f"起始值: {start}, 终止值: {end}")

        # 找出不在 segment 中的值，标记为 'n_segments'
        for time in a_times:
            found = False
            for segment in segments:
                start, end = segment
                if start <= time <= end:
                    found = True
                    break
            if not found:
                n_segments.append(time)

        # 打印不在 segment 中的 'a_times'
        for time in n_segments:
            print(f"'a_times' 中不在 'segments' 中的值: {time}")

        merged_onsets = []
        merged_durations = []

        for start, end in segments:  # 事件
            duration = end - start + 1  # 眨眼伪迹的持续时间（秒）

            merged_onsets.append(start)
            merged_durations.append(duration)

        merged_onsets.extend(n_segments)
        merged_onsets = [i + st for i in merged_onsets]
        for time in n_segments:
            merged_durations.append(1)
        # list, list, str
        merged_annotation = Annotations(onset=merged_onsets, duration=merged_durations, description=description)
        all_merged_annotations.append(merged_annotation)

    # 将所有的注释合并为一个
    merged_annotations = Annotations(onset=[], duration=[], description=[])
    for merged_annotation in all_merged_annotations:
        merged_annotations += merged_annotation
    # 将所有的注释应用于原始数据并绘制EEG保存
    raw.set_annotations(merged_annotations)

    # Plot
    eeg_plot = plot_raw(raw, duration=11, scalings=300e-6, show=False, show_scrollbars=False, start=st)
    # bgcolor = rcParams['axes.facecolor']
    # 获取第一个轴对象，通常包含图表的标题和注释
    ax = eeg_plot.mne.ax_main
    # 修改纵轴刻度字体属性
    for label in ax.yaxis.get_ticklabels():
        label.set_color(ThemeColorConfig.get_txt_color())  # 字体颜色
        label.set_fontsize(11)  # 字体大小
    # 遍历这个轴上的所有文本对象，设置字体
    for text in ax.texts:
        text.set_weight('bold')
        text.set_fontsize(11)
    # Add legend
    event_label = 'EB: Eye Blinking, FE: Frontal EMG, CE: Chew EMG, TE: Temporal EMG'
    # 获取了绘图的第一个轴（Axes）对象并设置图例
    eeg_plot.get_axes()[0].legend([event_label], loc='lower center', bbox_to_anchor=(0.35, -0.12), frameon=False,
                                  prop={'family': 'Arial', 'size': 12})

    # Save the figure
    eeg_plot.tight_layout()
    buffer = BytesIO()
    eeg_plot.savefig(buffer, format='png', dpi=600)
    buffer.seek(0)
    res_signal.emit(buffer.getvalue())

    annotations_list = []
    if auto:
        if len(merged_annotations.onset) > 0:
            for onset, duration, des in zip(merged_annotations.onset, merged_annotations.duration,
                                            merged_annotations.description):
                annotations_list.append([onset, duration, des])

    return annotations_list, des_list
