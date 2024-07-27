from torch import device, cuda, from_numpy, max, float32, no_grad, load
from numpy import array, mean, var, max as np_max
import HFO.model as mm
from utils.config import AddressConfig
from mne import Annotations
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph import mkPen, mkBrush, GraphicsView, GraphicsLayout, InfiniteLine, TextItem, LinearRegionItem

# import mne
# import pyqtgraph as pg

use_device = device("cuda" if cuda.is_available() else "cpu")


def hfo_process(raw, ch_idx, start_time, end_time):
    model = load_model()
    raw.pick(ch_idx)
    raw_notch, sliced_data = preprocess(raw, start_time, end_time)
    # show_plot(merged(raw_notch, predicted(model, sliced_data), start_time),start_time)  <test>
    return merged(raw_notch, predicted(model, sliced_data), start_time)


def load_model():
    model = mm.Celestial()
    # model_path = 'model_weights.pth'  <test>
    model_path = AddressConfig.get_hfo_adr('cp')
    model_weights = load(model_path, map_location=use_device)
    # 先把模型转移到正确的设备然后加载状态字典
    model.to(use_device).load_state_dict(model_weights)
    model.eval()
    return model


def preprocess(raw, start_time, end_time):
    raw.notch_filter(freqs=50)

    t_idx = raw.time_as_index([start_time, end_time])  # 将时间值转换为对应的数据索引 ndarray [5000, 10000]
    data, _ = raw[:, t_idx[0]:t_idx[1]]  # 从原始数据中提取指定时间段内的数据 (1,5000)
    data = data * 1e6
    sliced_data = []  # 99*(1,100)
    start_idx = 0
    while start_idx + 100 <= data.shape[1]:  # 0-100 50-150 ... 4900-5000: 99段 ——> 1段（0.1s） ——> 0-0.1 0.05-0.15 ...
        slice_data = data[:, start_idx:start_idx + 100]  # (1,100)
        sliced_data.append(slice_data)
        start_idx += 50  # 滑动窗口
    sliced_data = array(sliced_data)
    return raw, sliced_data  # (99,1,100)


def predicted(model, preprocessed_data):
    input_data = from_numpy(preprocessed_data).to(use_device)  # (99,1,100)
    num_samples = input_data.size(0)  # 99
    ones_indices = []  # 存储预测结果为1的序列号
    batch_size = 32
    for i in range(0, num_samples, batch_size):
        batch = input_data[i:i + batch_size]  # 获取当前小批量数据 (32,1,100)

        if i + batch_size > num_samples:
            # 处理最后一个不满 32 的批次
            remaining_samples = num_samples - i
            batch = input_data[i:i + remaining_samples]

        with no_grad():
            batch = batch.type(float32)
            output = model(batch)  # (32,2) remain(3,2)
            _, predicted_labels = max(output, 1)  # (32,) (3,)
            # 记录预测结果为1和0的序列号
            for j, label in enumerate(predicted_labels):
                if label == 1:
                    ones_indices.append(i + j)
    # print(ones_indices)
    return ones_indices


def merged(raw, predicted_data, start_time):
    intervals = [(0.05 * n + start_time, 0.05 * n + 0.1 + start_time) for n in predicted_data]
    merged_data = []
    for start, end in intervals:
        if not merged_data or merged_data[-1][1] < start:
            merged_data.append([start, end])
        else:
            merged_data[-1][1] = end

    descriptions = ['HFO'] * len(merged_data)
    onsets = [item[0] for item in merged_data]
    durations = [item[1] - item[0] for item in merged_data]
    raw.set_annotations(Annotations(onsets, durations, descriptions))
    return raw


def get_bias(high_data):
    tmp_data = high_data * 1e6
    bias = mean(tmp_data) + 2 * var(tmp_data)
    return bias * 1e-6


def yAxisTickFormatter(values, scale, spacing):
    """
    Custom tick formatter for y-axis to display in microvolts
    """
    return [f'{value * 1e6:.2f}' for value in values]


def show_plot(merged_raw, start_time):
    raw_low = merged_raw.copy().filter(l_freq=1, h_freq=70)
    raw_high = merged_raw.copy().filter(l_freq=80, h_freq=450)
    data1, times1 = raw_low[:]
    data2, times2 = raw_high[:]
    data, time = merged_raw[:]

    # app = pg.mkQApp('123')  # <test>
    view = GraphicsView()
    layout = GraphicsLayout(border=(255, 255, 255))
    view.setCentralItem(layout)
    # view.setMinimumSize(400, 300)  # 设置最小宽度为400px，最小高度为300px
    # view.resize(800, 600)
    view.setBackground('w')

    # 创建三个绘图区域
    layout.setContentsMargins(10, 10, 10, 10)
    p1 = layout.addPlot(row=0, col=0)
    p1.setTitle("<span style='color: black; font-weight: bold; font-size: 12pt;'>raw</span>")
    p2 = layout.addPlot(row=1, col=0)
    p2.setTitle("<span style='color: black; font-weight: bold; font-size: 12pt;'>1-70Hz</span>")
    p3 = layout.addPlot(row=2, col=0)
    p3.setTitle("<span style='color: black; font-weight: bold; font-size: 12pt;'>80-450Hz</span>")
    p1.setXLink(p2)  # 链接两个图的x轴
    p2.setXLink(p3)

    # Set axis label, width and tick colors
    axis_pen = mkPen(color='k', width=2)  # Black color for axis and ticks

    # Set axis labels and disable SI prefix
    label_style = {'font-size': '12pt'}
    tick_font = QtGui.QFont('Arial', 11)

    for plot in [p1, p2, p3]:
        plot.getAxis('left').setPen(axis_pen)
        plot.getAxis('left').setTextPen(axis_pen)
        plot.getAxis('bottom').setPen(axis_pen)
        plot.getAxis('bottom').setTextPen(axis_pen)
        plot.getAxis('left').setWidth(100)
        # plot.hideButtons()  # hide show all

        # Apply custom tick formatter to y-axis
        plot.getAxis('left').tickStrings = yAxisTickFormatter
        # Set axis labels and custom tick formatter
        plot.getAxis('left').setStyle(tickFont=tick_font, tickTextOffset=8)
        plot.getAxis('bottom').setStyle(tickFont=tick_font)
        plot.getAxis('left').setLabel('Amplitude (μV)', **label_style)
        plot.getAxis('bottom').setLabel('Time', units='s', **label_style)
        plot.getAxis('left').enableAutoSIPrefix(False)

    # Plot
    p1.plot(time, data[0], pen=mkPen(color='royalblue'))
    p2.plot(times1, data1[0], pen=mkPen(color='royalblue'))
    p3.plot(times2, data2[0], pen=mkPen(color='royalblue'))

    # Set x range to show
    for plot in [p1, p2, p3]:
        plot.setXRange(start_time, start_time + 5)

    # l2.layout.setRowStretchFactor(0, 1)  # Stretch factor for p21
    # l2.layout.setRowStretchFactor(1, 1)  # Stretch factor for p23

    # Add bias
    bias = get_bias(data2)
    hline_p3_low = InfiniteLine(pos=-bias, angle=0, pen=mkPen(color='r', style=QtCore.Qt.DashLine))
    hline_p3_high = InfiniteLine(pos=bias, angle=0, pen=mkPen(color='r', style=QtCore.Qt.DashLine))
    p3.addItem(hline_p3_low)
    p3.addItem(hline_p3_high)

    # Add annotations to the plot
    for annotation in merged_raw.annotations:
        onset = annotation['onset']
        duration = annotation['duration']
        description = annotation['description']

        # Add text at the onset time
        text_p2 = TextItem('Spike', anchor=(0.5, 1), color='r')
        text_p3 = TextItem(description, anchor=(0.5, 1), color='r')
        font = QtGui.QFont()  # 设置字体
        font.setBold(True)
        font.setPointSize(14)  # Set the font size
        text_p2.setFont(font)
        text_p3.setFont(font)
        text_p2.setPos(onset + duration / 2, np_max(data1[0]))  # Set the position of the text
        text_p3.setPos(onset + duration / 2, np_max(data2[0]))  # Set the position of the text
        p2.addItem(text_p2)
        p3.addItem(text_p3)

        # Add region
        region_p1 = LinearRegionItem(values=(onset, onset + duration), brush=mkBrush(135, 206, 250, 50),
                                     movable=False)
        region_p2 = LinearRegionItem(values=(onset, onset + duration), brush=mkBrush(135, 206, 250, 50),
                                     movable=False)
        region_p3 = LinearRegionItem(values=(onset, onset + duration), brush=mkBrush(135, 206, 250, 50),
                                     movable=False)
        p1.addItem(region_p1)
        p2.addItem(region_p2)
        p3.addItem(region_p3)

    # view.show()  # <test>
    # pg.exec()  # <test>

    return view

# if __name__ == '__main__':
#     hfo_process(mne.io.read_raw("../test_data/HFO/管若彤_filtered.edf", preload=True), 11, 5, 10)
