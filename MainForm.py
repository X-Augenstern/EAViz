from sys import argv, exit
from PyQt5 import uic
from PyQt5.QtGui import QPalette, QPixmap, QBrush
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib import use
from matplotlib.pyplot import rcParams, close, subplots, tight_layout
from numpy import atleast_1d
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from SelectSignalsForm import SelectSignalsForm
from FilterOptionsForm import FilterOptionsForm
from SelectSignalsForPSD import SelectSignalsForPSD
from SelectTimeSpanForTPM import SelectTimeSpanForTPM
from ExtendServicesForm import ExtendServicesForm
from ExportForm import ExportForm
from utils.filter_info import FilterInfo
from utils.eeg_plot_info import EEGPlotInfo
from utils.config import ChannelConfig, AddressConfig
from pyqtgraph.dockarea import DockArea
from utils.custom_widgets import CanvasDock, SaveDock
from qdarkstyle import load_stylesheet_pyqt5, load_stylesheet, LightPalette
from utils.diary import Diary
from mne import Annotations, set_config
from warnings import filterwarnings

use('Qt5Agg')
set_config('MNE_BROWSER_BACKEND', 'matplotlib')
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
filterwarnings("ignore", category=UserWarning,
               message="Starting a Matplotlib GUI outside of the main thread will likely fail.")
ui_main_form = uic.loadUiType('ui/main_form.ui')[0]


# set_config('MNE_BROWSER_BACKEND', 'qt')  # mne-qt-browser

class MainForm(QFrame, ui_main_form):
    def __init__(self):
        super(MainForm, self).__init__()
        self.diary = Diary(log_folder_path=AddressConfig.log_folder_path).init_logger()
        self.diary.debug('START')
        self.setupUi(self)
        self.init_ui()
        self.adr = None
        self.select_signal_form = None
        self.eeg_list = None
        self.filter_options_form = None
        self.extend_services_form = None
        self.fi = FilterInfo()
        self.epi = None
        self.raw = None
        self.saved_raw = None
        self.dock_area = DockArea(self)
        self.gridLayout.addWidget(self.dock_area)
        self.eeg_dock = CanvasDock('eeg', self.dock_area)
        self.dock_area.addDock(self.eeg_dock)
        self.psd_dock = None
        self.tpm_dock = None
        self.select_signals_for_psd = None
        self.select_time_span_for_tpm = None
        self.channel_config = ChannelConfig()
        self.export_form = None
        self.t_max = None  # 最后一次sample的开始时间
        self.groupbox_list = [self.groupBox_1, self.groupBox_2, self.groupBox_3, self.groupBox_4, self.groupBox_5]

    def init_ui(self):
        # self.update_background('C:/Users/hp/Desktop/1.png')
        self.change_size_cbx.setCurrentIndex(1)
        self.select_file_btn.clicked.connect(self.load_signals)
        self.change_signal_btn.clicked.connect(self.change_signals)
        self.change_filter_btn.clicked.connect(self.load_filter)
        self.filter_cbx.clicked.connect(self.choose_filter)
        self.psd_btn.clicked.connect(self.show_psd)
        self.psd_switch_cbx.clicked.connect(self.close_psd)
        self.tpm_btn.clicked.connect(self.show_tpm)
        self.tpm_switch_cbx.clicked.connect(self.close_tpm)
        self.increase_amp_btn.clicked.connect(lambda: self.change_amplitude('p'))
        self.decrease_amp_btn.clicked.connect(lambda: self.change_amplitude('m'))
        self.custom_amp_btn.clicked.connect(lambda: self.change_amplitude('c'))
        # 默认会传递当前选中项的索引值，所以custom要显式传递
        self.change_size_cbx.activated.connect(lambda: self.change_window_size(False))
        self.custom_size_btn.clicked.connect(lambda: self.change_window_size(True))
        self.increase_signal_btn.clicked.connect(lambda: self.change_num_signals('p'))
        self.decrease_signal_btn.clicked.connect(lambda: self.change_num_signals('m'))
        self.theme_btn.clicked.connect(lambda: change_theme(self.theme_btn.text()))
        self.jmp_btn.clicked.connect(self.jump_to)
        self.fs_btn.clicked.connect(self.full_screen)
        self.export_btn.clicked.connect(self.export_edf)
        self.es_btn.clicked.connect(self.extend_services)
        self.listWidget.itemClicked.connect(self.show_ann)
        self.listWidget.itemDoubleClicked.connect(self.jump_to_ann)
        self.rename_btn.clicked.connect(self.rename_ann)
        self.delete_btn.clicked.connect(self.delete_ann)
        self.create_btn.clicked.connect(self.create_ann)
        self.duration.valueChanged.connect(self.autoset_end)
        self.start_time.valueChanged.connect(self.autoset_end)
        self.save_txt_btn.clicked.connect(self.save_ann)
        self.clear_btn.clicked.connect(self.clear_ann)

    def load_signals(self):
        """
        Load SelectSignalsForm.
        """
        adr, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '*.edf')
        if adr is None or len(adr) == 0:
            return
        else:
            self.select_signal_form = SelectSignalsForm(self, adr)
            self.select_signal_form.show()
            self.diary.debug('MainForm —> SelectSignalsForm')

    def ini_before_load(self):
        """
        When a new raw loaded or signal changed, close the ExtendServiceForm and PSD / TPM Dock if exists.
        """
        self.filter_cbx.setChecked(False)  # 防止回退到原先的raw
        self.filter_cbx.setText('on')
        if self.extend_services_form:
            self.extend_services_form.close()
            self.extend_services_form = None
        if self.psd_dock is not None:
            self.psd_switch_cbx.setChecked(False)
            self.close_psd()
        if self.tpm_dock is not None:
            self.tpm_switch_cbx.setChecked(False)
            self.close_tpm()

    def get_raw(self, raw, eeg_list, adr=None, preload=False):
        """
        根据所选channels绘制相应的eeg图

        导入和改变通道的区别在于是否重置：地址 + EEG绘图参数 + 获取事件

        但是二者都是从原始的edf中读取信号，因此改变通道后滤波需要重新滤，如果直接基于滤波后的raw改变通道，多改少没问题，但是少改多又要重新从原始
        edf中读取，又会丢失滤波信息
        :param preload: Use to distinguish the function Select file or Change signals.
        """
        self.ini_before_load()
        self.raw = raw
        self.eeg_list = eeg_list

        if not preload:
            self.adr = adr
            text = 'Plotting: ' + adr.split('/')[-1]
            self.label.setToolTip(adr.split('/')[-1])
            if len(text) > 37:
                text = text[:34] + '...'
            self.label.setText(text)

            self.epi = EEGPlotInfo(len(eeg_list))
            self.change_size_cbx.setCurrentIndex(1)

            self.get_ann()
            self.t_max = self.raw.times[-1]
            self.end_time.setMaximum(self.t_max)
            self.start_time.setMaximum(self.t_max)
            self.duration.setMaximum(self.t_max)

        self.plot_eeg(raw)
        self.filter_cbx.setEnabled(True)

    def plot_eeg(self, raw, start=0.0, epi_check=False):
        """
        根据raw绘制相应的eeg图
        """
        # duration：window内采样时间
        # n_channels：window内channel数
        # matplotlib backend
        if epi_check:
            if not self.epi.info_changed:
                return
        eeg_plot = FigureCanvas(raw.plot(duration=self.epi.window_size, start=start, n_channels=self.epi.n_channels,
                                         scalings=self.epi.amplitude, show=False, use_opengl=True,
                                         color='royalblue'))  # 绑定Figure到Canvas上
        if self.epi.info_changed:
            self.epi.info_changed = False
        # plt.tight_layout()
        close()  # More than 20 figures have been opened.
        eeg_plot.setFocusPolicy(Qt.StrongFocus)  # 将焦点策略设置为Qt.StrongFocus，以便接收键盘事件
        eeg_plot.mpl_connect('key_press_event', self.keyPressEvent)  # mpl_connect方法将键盘事件连接到canvas上
        self.eeg_dock.change_canvas(eeg_plot)

        # qt backend
        # eeg_plot = raw.plot(duration=self.window_size, start=start, n_channels=10,
        #                     scalings=300e-6 * self.scale_factor_percent_list[self.scale_index],
        #                     show=False, use_opengl=True, color='royalblue')
        # eeg_dock = Dock('eeg', widget=eeg_plot)
        # self.dock_area.addDock(eeg_dock)

        self.diary.debug('Plot EEG')

    def change_signals(self):
        """
        Load SelectSignalsForm.
        """
        if self.adr is None or len(self.adr) == 0:
            return
        else:
            self.select_signal_form = SelectSignalsForm(self, self.adr, preload=True)
            self.select_signal_form.show()
            self.diary.debug('MainForm —> SelectSignalsForm')

    def load_filter(self):
        """
        Load FilterOptionsForm
        """
        self.filter_options_form = FilterOptionsForm(self)
        self.filter_options_form.show()
        self.diary.debug('MainForm —> FilterOptionsForm')

    def choose_filter(self):
        """
        根据所选filter进行相应的滤波
        """
        if self.filter_cbx.isChecked():
            self.saved_raw = self.raw  # save last step
        else:
            self.raw = self.saved_raw
            self.diary.debug('Revert to the saved RAW')
            self.plot_eeg(self.raw)
            self.filter_cbx.setText('on')
            return

        diary_msg = 'Filtered Message:'
        # notch
        if self.fi.do_notch == 1:
            raw_filtered = self.use_filter(self.raw.copy(), notch=1)
            diary_msg += f'\nnotch: {self.fi.notch}Hz'
        # without notch
        else:
            raw_filtered = self.raw.copy()

        # lp
        if self.fi.do_lp == 1 and self.fi.do_hp == 0:
            raw_filtered = self.use_filter(raw_filtered, lp=1)
            diary_msg += f'\nlp: {self.fi.hf}Hz'
        # hp
        elif self.fi.do_lp == 0 and self.fi.do_hp == 1:
            raw_filtered = self.use_filter(raw_filtered, hp=1)
            diary_msg += f'\nhp: {self.fi.lf}Hz'
        # bp
        elif self.fi.do_bp == 1:
            raw_filtered = self.use_filter(raw_filtered, bp=1)
            diary_msg += f'\nbp: {self.fi.bp_lf} - {self.fi.bp_hf}Hz'
        # lp + hp = bp
        elif self.fi.do_lp == 1 and self.fi.do_hp == 1:
            raw_filtered = self.use_filter(raw_filtered, lp=1, hp=1)
            diary_msg += f'\nbp: {self.fi.lf} - {self.fi.hf}Hz'

        self.raw = raw_filtered
        self.diary.info(diary_msg)
        self.plot_eeg(raw_filtered)
        self.filter_cbx.setText('off')

    def use_filter(self, raw_to_filter, notch=0, lp=0, hp=0, bp=0):
        """
        滤波
        """
        raw = raw_to_filter.copy()
        if notch == 1:  # notch
            print("------------------------------------------using notch------------------------------------------")
            notch_freq = self.fi.notch  # 要滤除的频率（以Hz为单位）
            raw.notch_filter(freqs=notch_freq)
        elif lp == 1 and hp == 1:  # bandpass
            print("----------------------------------------using bandpass----------------------------------------")
            low_freq = self.fi.lf
            high_freq = self.fi.hf
            raw.filter(l_freq=low_freq, h_freq=high_freq)
        elif lp == 1:  # lowpass
            print("----------------------------------------using lowpass----------------------------------------")
            high_freq = self.fi.hf
            raw.filter(l_freq=None, h_freq=high_freq)
        elif hp == 1:  # highpass
            print("----------------------------------------using highpass----------------------------------------")
            low_freq = self.fi.lf
            raw.filter(l_freq=low_freq, h_freq=None)
        elif bp == 1:  # bandpass
            print("----------------------------------------using bandpass----------------------------------------")
            low_freq = self.fi.bp_lf
            high_freq = self.fi.bp_hf
            raw.filter(l_freq=low_freq, h_freq=high_freq)
        return raw

    def show_psd(self):
        """
        Load SelectSignalsForPSD.
        """
        if self.raw is None:
            return
        self.select_signals_for_psd = SelectSignalsForPSD(self)
        self.select_signals_for_psd.show()
        self.diary.debug('MainForm —> SelectSignalsForPSD')

    def plot_psd(self, para_list=None):
        """
        绘制PSD
        """
        num_plots = len(para_list[0])
        fig, axs = subplots(num_plots, 1, figsize=(10, num_plots * 2.5))  # 创建多个子图，固定Figure高度(宽,高 inch)
        # <FAILED>'Axes' object is not subscriptable
        # 当 len(para_list[0]) 为1时，plt.subplots 返回的是一个单独的 Axes 对象，而不是一个包含多个 Axes 对象的数组
        axs = atleast_1d(axs)  # 确保 axs 始终是一个数组，即使只有一个子图时也是如此
        for i, ch_name in enumerate(para_list[0]):
            color = ChannelConfig.colors[i % len(ChannelConfig.colors)]
            self.raw.compute_psd(fmin=para_list[1], fmax=para_list[2], picks=ch_name).plot(
                axes=axs[i], color=color, spatial_colors=False, dB=True, amplitude=False, show=False)
            axs[i].set_title(f'PSD for {ch_name}')
        tight_layout()  # 调整子图布局

        if self.psd_dock is not None:
            self.psd_dock.close()
            self.psd_dock = None

        psd_plot = FigureCanvas(fig)
        psd_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        psd_plot.setFixedHeight(num_plots * 250)  # 每个子图固定高度

        self.psd_dock = SaveDock('psd', area=self.dock_area, canvas=psd_plot, hideTitle=True, enable_scroll=True)
        self.dock_area.addDock(self.psd_dock, 'bottom', self.eeg_dock)

        if not self.psd_switch_cbx.isEnabled():
            self.psd_switch_cbx.setEnabled(True)
            self.psd_switch_cbx.setChecked(True)
            self.psd_switch_cbx.setText('off')
        self.diary.debug('Plot PSD')

    def close_psd(self):
        """
        关闭psd绘图
        """
        self.psd_dock.close()
        self.psd_dock = None
        self.psd_switch_cbx.setEnabled(False)
        self.psd_switch_cbx.setText('on')
        self.diary.debug('Close PSD')

    def show_tpm(self):
        """
        Load SelectTimeSpanForTPM.
        """
        if self.raw is None:
            return
        self.select_time_span_for_tpm = SelectTimeSpanForTPM(self)
        self.select_time_span_for_tpm.show()
        self.diary.debug('MainForm —> SelectTimeSpanForTPM')

    def plot_tpm(self, span_list=None):
        """
        绘制topomap图
        """
        intersection = set(self.channel_config.topomap_channels) & set(self.eeg_list)
        if not intersection:
            QMessageBox.warning(self, 'Warning', 'The selected channel(s) is(are) not included in TPM channels!')
            return

        fig, axes = subplots(1, 5)
        raw = self.raw.copy().pick(picks=list(intersection))
        raw.set_montage(self.channel_config.get_montage())

        tpm_plot = None
        tpm_title = None

        try:
            if span_list is not None:
                # tpm_plot = FigureCanvas(
                #     raw.compute_psd(fmin=0, fmax=45, tmin=span_list[0], tmax=span_list[1], picks='all').plot_topomap(
                #         dB=True, show=False, show_names=lambda x: x.replace(f'{self.channel_config.prefix}', ''),
                #         axes=axes))
                tpm_plot = FigureCanvas(
                    raw.compute_psd(fmin=0, fmax=45, tmin=span_list[0], tmax=span_list[1], picks='all').plot_topomap(
                        dB=True, ch_type='eeg', show=False, show_names=True, axes=axes))  # vlim='joint' 共用一个colorbar
                # for ax in axes:
                #     ax.set_title(ax.get_title(), fontsize=14, y=1.1)  # 设置每个子图的标题、字体大小、高度
                tpm_title = f'{span_list[0]}-{span_list[1]}s'
            else:
                # tpm_plot = FigureCanvas(
                #     raw.compute_psd(fmin=0, fmax=45, picks='all').plot_topomap(
                #         dB=True, show=False, show_names=lambda x: x.replace(f'{self.channel_config.prefix}', ''),
                #         axes=axes))
                tpm_plot = FigureCanvas(
                    raw.compute_psd(fmin=0, fmax=45, picks='all').plot_topomap(
                        dB=True, ch_type='eeg', show=False, show_names=True, axes=axes))
                tpm_title = f'{raw.times.min()}-{raw.times.max()}s'
            tight_layout()
            fig.subplots_adjust(left=0.01, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.2)
        except Exception as e:
            self.diary.error('<FAILED>' + str(e))
            QMessageBox.warning(self, 'Warning', 'The count of valid channels should be more than 2!')

        if tpm_plot is not None:
            if self.tpm_dock is not None:
                self.tpm_dock.close()
                self.tpm_dock = None

            self.tpm_dock = SaveDock('topomap', area=self.dock_area, canvas=tpm_plot, size=(10, 5), fig_title=tpm_title)
            self.dock_area.addDock(self.tpm_dock, 'top', self.eeg_dock)

            if not self.tpm_switch_cbx.isEnabled():
                self.tpm_switch_cbx.setEnabled(True)
                self.tpm_switch_cbx.setChecked(True)
                self.tpm_switch_cbx.setText('off')
        self.diary.debug('Plot TPM')

    def close_tpm(self):
        """
        关闭tpm绘图
        """
        self.tpm_dock.close()
        self.tpm_dock = None
        self.tpm_switch_cbx.setEnabled(False)
        self.tpm_switch_cbx.setText('on')
        self.diary.debug('Close TPM')

    def change_amplitude(self, flag):
        """
        放大/缩小/自定义信号幅度
        :param flag: 'p'(plus) | 'm'(minus) | 'c'(custom)
        """
        if self.raw is None:
            return
        if flag == 'p':
            self.epi.minus_scale_idx()
        elif flag == 'm':
            self.epi.plus_scale_idx()
        elif flag == 'c':
            self.epi.set_base_amp(self.custom_amp.value())
        self.plot_eeg(self.raw, epi_check=True)
        self.diary.info(f'scale: {self.epi.amplitude}')

    def change_window_size(self, custom):
        """
        改变窗口尺寸
        """
        if self.raw is None:
            return
        if not custom:
            self.epi.set_window_size(int(self.change_size_cbx.currentText()[:-1]))
        else:
            if self.custom_size.value() >= self.t_max:
                self.custom_size.setValue(self.t_max)
            self.epi.set_window_size(self.custom_size.value())
        self.plot_eeg(self.raw, epi_check=True)
        self.diary.info(f'window size: {self.epi.window_size}')

    def change_num_signals(self, flag):
        """
        增加/减少window内channel数
        :param flag: 'p'(plus) | 'm'(minus)
        """
        if self.raw is None:
            return
        if flag == 'p':
            self.epi.plus_n_channels()
        elif flag == 'm':
            self.epi.minus_n_channels()
        self.plot_eeg(self.raw, epi_check=True)
        self.diary.info(f'channel nums: {self.epi.n_channels}')

    def jump_to(self):
        """
        EEG图跳转至指定时间
        """
        if self.raw is None:
            return
        if self.jmp_time.value() >= self.t_max:
            self.jmp_time.setValue(self.t_max)
        self.plot_eeg(self.raw, start=self.jmp_time.value())

    def full_screen(self):
        """
        EEG图全屏
        """
        if self.raw is None:
            return
        for gb in self.groupbox_list:
            gb.setVisible(False)
        super().showFullScreen()

    def exit_full_screen(self):
        """
        退出EEG图全屏
        """
        self.showNormal()
        for gb in self.groupbox_list:
            gb.setVisible(True)

    def export_edf(self):
        """
        Load ExportForm.
        """
        if self.raw is None:
            return
        self.export_form = ExportForm(self)
        self.export_form.show()
        self.diary.debug('MainForm —> ExportForm')

    def extend_services(self):
        """
        Load ExtendServicesForm.
        """
        if self.extend_services_form is not None:
            self.extend_services_form.setWindowState(Qt.WindowNoState)  # 取消最小化状态
            self.extend_services_form.activateWindow()  # 激活并带至前台
            return
        self.extend_services_form = ExtendServicesForm(self)
        self.extend_services_form.show()
        self.diary.debug('MainForm —> ExtendServicesForm')

    def get_ann(self):
        """
        获取事件
        """
        ann_list = self.raw.annotations.description
        self.listWidget.clear()
        self.listWidget.addItems(ann_list)

    def show_ann(self):
        """
        展示当前事件相关信息
        """
        self.ann_txt.setText(self.listWidget.currentItem().text())
        ann_idx = self.listWidget.currentRow()
        if ann_idx >= 0:
            selected_ann = self.raw.annotations[ann_idx]
            start_time = selected_ann['onset']
            duration = selected_ann['duration']
            self.start_time.setValue(start_time)
            self.end_time.setValue(start_time + duration)
            self.duration.setValue(duration)

    def jump_to_ann(self):
        """
        跳转至当前事件处
        """
        ann_idx = self.listWidget.currentRow()
        if ann_idx >= 0:
            self.plot_eeg(self.raw, start=self.raw.annotations[ann_idx]['onset'])

    def create_ann(self):
        """
        添加事件
        """
        if self.raw is None:
            return
        if self.ann_txt.text() == '':
            return
        description = self.ann_txt.text()
        onset = self.start_time.value()
        duration = self.duration.value()
        self.diary.info('Create Ann:\n'
                        f'onset: {onset}\n,'
                        f'duration: {duration}\n'
                        f'des: {description}\n'
                        f'Max time: {self.t_max}')
        self.raw.annotations.append(onset, duration, description)
        self.plot_eeg(self.raw, start=onset)
        self.get_ann()

    def rename_ann(self):
        """
        更新事件名
        """
        if self.raw is None:
            return
        ann_idx = self.listWidget.currentRow()
        if ann_idx >= 0:
            description = self.ann_txt.text()
            # self.raw.annotations.description[ann_idx] = description  可能会失败
            onset = self.raw.annotations.onset[ann_idx]
            duration = self.raw.annotations.duration[ann_idx]
            self.diary.info(f'Rename Ann: {self.raw.annotations.description[ann_idx]} to {description}')
            self.raw.annotations.delete(ann_idx)
            self.raw.annotations.append(onset, duration, description)
            self.plot_eeg(self.raw, start=self.raw.annotations[ann_idx]['onset'])
            self.get_ann()

    def delete_ann(self):
        """
        删除事件
        """
        if self.raw is None:
            return
        ann_idx = self.listWidget.currentRow()
        if ann_idx >= 0:
            onset = self.raw.annotations[ann_idx]['onset']
            self.diary.info(f'Delete Ann: {self.raw.annotations.description[ann_idx]}')
            self.raw.annotations.delete(ann_idx)
            self.plot_eeg(self.raw, start=onset)
            self.get_ann()

    def clear_ann(self):
        """
        清空事件
        """
        if self.raw is None:
            return
        self.diary.info('Clear Ann')
        self.raw.set_annotations(Annotations([], [], []))
        self.plot_eeg(self.raw)
        self.get_ann()

    def autoset_end(self):
        """
        控制时间
        """
        if self.raw is None:
            return
        if self.end_time.value() == self.t_max:
            return
        if self.start_time.value() + self.duration.value() > self.t_max:
            self.end_time.setValue(self.t_max)
            self.duration.setValue(self.end_time.value() - self.start_time.value())
        self.end_time.setValue(self.start_time.value() + self.duration.value())

    def save_ann(self):
        """
        保存事件为.txt
        """
        if self.raw is None:
            return
        adr, _ = QFileDialog.getSaveFileName(self, 'Save .txt', '.', 'TXT files (*.txt)')
        if adr is None or len(adr) == 0:
            return
        else:
            # self.raw.annotations.save(adr, overwrite=True)  # 未提供utf-8接口
            with open(adr, 'w', encoding='utf-8') as file:
                file.write('# onset, duration, description\n')
                for onset, duration, description in zip(self.raw.annotations.onset, self.raw.annotations.duration,
                                                        self.raw.annotations.description):
                    line = f'{onset},{duration},{description}\n'
                    file.write(line)
            self.diary.info('Save Annotations')

    def resizeEvent(self, event):
        """
        窗口尺寸变化时更新背景
        """
        # self.update_background('C:/Users/hp/Desktop/1.png')

    def update_background(self, img_path):
        """
        根据窗口尺寸调整背景尺寸
        """
        pix = QPixmap(img_path)
        pix = pix.scaled(self.size(), aspectRatioMode=Qt.KeepAspectRatioByExpanding)
        # background
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pix))
        self.setPalette(palette)

    def keyPressEvent(self, event):
        """
        键盘事件处理函数
        """
        if event.key == 'escape':
            if self.isFullScreen():
                self.exit_full_screen()

    def closeEvent(self, event):
        """
        程序关闭事件
        """
        reply = QMessageBox.question(self, 'Quit', "Are you sure?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.diary.debug('EXIT')
            if self.extend_services_form:
                self.extend_services_form.close()
            if self.tpm_dock is not None:
                self.tpm_dock.close()
            self.close()
            event.accept()
        else:
            event.ignore()


def change_theme(theme):
    """
    改变界面外观
    """
    if theme == 'Light':
        # app.setStyleSheet('')
        app.setStyleSheet(load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
        main_form.theme_btn.setText('Dark')
    if theme == 'Dark':
        app.setStyleSheet(load_stylesheet_pyqt5())
        main_form.theme_btn.setText('Light')


if __name__ == "__main__":
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # 启用高DPI缩放
    app = QApplication(argv)
    app.setStyleSheet(load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    main_form = MainForm()
    main_form.show()
    exit(app.exec_())
