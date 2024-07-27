from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
import threading
import sys
import numpy as np
import torch
import scipy.io as scio
import offline_process as ofp
from A3D_model import R3DClassifier
import plot_result as pr

ultra_v3 = uic.loadUiType('ui/ultra_v3.ui')[0]

# stop_plot_Event = threading.Event()
# stop_plot_Event.clear()  # 创建关闭事件并设为未触发


class EpilepsySyndromeAnalyseForm(QMainWindow, ultra_v3):

    def __init__(self, parent=None):
        super(EpilepsySyndromeAnalyseForm, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()

        self.offline_start_index = 1
        self.data = np.zeros((21, 1), np.dtype('float32'))
        self.isstart = True
        self.offline_isstart = True
        self.isstop = False
        self.offline_isstop = False
        self.cur_index = 0

        self.model = R3DClassifier(7, (2, 2, 2, 2), pretrained=True)
        checkpoint = torch.load(
            'E:\\Project1\\A3D-EEG_epoch-1.pth.tar',
            map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.labels = ['BECT', 'CAE', 'CSWS', 'EIEE', 'FS', 'Normal', 'WEST']
        self.label = torch.as_tensor([6])
        self.montage = 'All'
        self.speed = 4000

        # self.plotTh = None

    def init_ui(self):
        self.import_data.clicked.connect(self.run_offline_import_data)
        self.offline_analyse.clicked.connect(self.run_offline_analyse)
        self.offline_stop.clicked.connect(self.run_offline_stop)

    def run_offline_import_data(self):
        datapath = QtWidgets.QFileDialog.getOpenFileName()
        try:
            self.data = scio.loadmat(datapath[0])['Raw_data']  # (21, 900001 ms)
            self.statusBar().showMessage('数据加载成功！')

        except BaseException:
            self.statusBar().showMessage('数据加载失败！')

    def run_offline_analyse(self):
        self.offline_isstart = True
        self.offline_isstop = False
        self.run_offline_plot(self.cur_index)

    def run_offline_stop(self):
        self.offline_isstart = False
        self.offline_isstop = True

    def run_offline_plot(self, index):
        if self.offline_isstart:
            for i in range(self.offline_start_index, self.data.shape[1] - 3999, self.speed):  # 0 - 896001
                img = ofp.read(self.data[:, i:(i + 3999)], self.montage)

                self.show_img(self.offline_signal,
                              "ESA\\offline_analyse\\signal" + f"\\{self.montage}.png")
                self.show_img(self.offline_filter_signal,
                              "ESA\\offline_analyse\\signal_filter" + f"\\{self.montage}.png")
                self.show_img(self.offline_feature,
                              "ESA\\offline_analyse\\feature" + f"\\{self.montage}.png")

                img = torch.tensor(img.astype('float32'))
                img = img.reshape(1, 1, 21, 32, 32)
                output = self.model(img)
                pr.plot_figure(output, 'ESA\\offline_analyse')
                self.show_img(self.offline_result, "ESA\\offline_analyse\\result\\" + "pro.png")
                if not self.isMaximized():
                    self.showMaximized()
                if self.offline_isstop:
                    self.cur_index = index
                    if i + self.speed < self.data.shape[1] - 3999:
                        self.offline_start_index = i + self.speed
                    break
        # self.plotTh = PlotThread(index, self.offline_start_index, self.data, self.model)
        # self.plotTh.plotSignal.connect(self.show_img)
        # self.plotTh.plotSignal1.connect(self.pr_plot)
        # self.plotTh.start()

    # def pr_plot(self, output):
    #     pr.plot_figure(output, 'EpilepsySyndromeAnalyse\\offline_analyse')
    #
    # def show_img(self):
    #     self.img_show(self.offline_signal,
    #                   "EpilepsySyndromeAnalyse\\offline_analyse\\signal" + f"\\{self.montage}.png")
    #     self.img_show(self.offline_filter_signal,
    #                   "EpilepsySyndromeAnalyse\\offline_analyse\\signal_filter" + f"\\{self.montage}.png")
    #     self.img_show(self.offline_feature,
    #                   "EpilepsySyndromeAnalyse\\offline_analyse\\feature" + f"\\{self.montage}.png")
    #     self.img_show(self.offline_result, "EpilepsySyndromeAnalyse\\offline_analyse\\result\\" + "pro.png")

    def show_img(self, label, img_path):
        redImg = QImage()
        QImage.load(redImg, img_path, format='png')
        label.setPixmap(QPixmap(redImg))
        QtWidgets.QApplication.processEvents()  # 动态更新图片


# # todo 专门为plot创建线程类
# class PlotThread(QThread):
#     plotSignal = pyqtSignal()
#     plotSignal1 = pyqtSignal(torch.Tensor)
#
#     def __init__(self, index, offline_start_index, data, model):
#         super(PlotThread, self).__init__()
#         self.index = index
#         self.speed = 4000
#         self.montage = 'All'
#         self.offline_start_index = offline_start_index
#         self.data = data
#         self.model = model
#
#     def plot(self):
#         for i in range(self.offline_start_index, self.data.shape[1] - 3999, self.speed):
#             img = ofp.read(self.data[:, i:(i + 3999)], self.montage)
#             img = torch.tensor(img.astype('float32'))
#             img = img.reshape(1, 1, 21, 32, 32)
#             output = self.model(img)
#             print(type(output))
#             self.plotSignal1.emit(output)
#             self.plotSignal.emit()
#             # if not self.isMaximized():
#             #     self.showMaximized()
#             # if self.offline_isstop:
#             #     self.cur_index = self.index
#             #     if i + self.speed < self.data.shape[1] - 3999:
#             #         self.offline_start_index = i + self.speed
#             #     break
#
#     def run(self):
#         self.plot()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_form = EpilepsySyndromeAnalyseForm()
    main_form.show()
    sys.exit(app.exec_())
