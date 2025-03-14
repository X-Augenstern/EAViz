from sys import argv, exit
from PyQt5.QtGui import QPalette, QPixmap, QBrush, QIcon, QColor
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib import use, style
from matplotlib.patheffects import Stroke, Normal
from matplotlib.pyplot import rcParams, subplots, tight_layout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numpy import atleast_1d
from SelectSignalsForm import SelectSignalsForm
from FilterOptionsForm import FilterOptionsForm
from SelectSignalsForPSD import SelectSignalsForPSD
from SelectTimeSpanForTPM import SelectTimeSpanForTPM
from ExtendServicesForm import ExtendServicesForm
from ExportForm import ExportForm
from utils.filter_info import FilterInfo
from utils.config import AddressConfig, PSDEnum, ChannelEnum, ThemeColorConfig
from utils.edf import EdfUtil
from utils.eeg_browser_extend import EEGBrowserManager
from utils.custom_widgets import QWidgetDock, SaveDock
from utils.diary import Diary
from pyqtgraph.dockarea import DockArea
from mne import set_config
from warnings import filterwarnings
from qdarkstyle import load_stylesheet_pyqt5, load_stylesheet, LightPalette
from qfluentwidgets import toggleTheme
from qfluentwidgets.components.dialog_box.color_dialog import ColorDialog
from ui.main_form_cur import Ui_Form
from os import path
import sys


use('Qt5Agg')
set_config('MNE_BROWSER_BACKEND', 'matplotlib')
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
filterwarnings("ignore", category=UserWarning,
               message="Starting a Matplotlib GUI outside of the main thread will likely fail.")
base_path = getattr(sys, '_MEIPASS', path.abspath('.'))
vd_path = path.join(base_path, 'VD')
sys.path.append(vd_path)

# ui_main_form = uic.loadUiType('ui/main_form_cur.ui')[0]


class MainForm(QFrame, Ui_Form):
    def __init__(self):
        super(MainForm, self).__init__()
        self.diary = Diary(log_folder_path=AddressConfig.log_folder_path).init_logger()
        self.diary.debug('START')
        self.setupUi(self)
        self.adr = None
        self.select_signal_form = None
        self.eeg_list = None
        self.filter_options_form = None
        self.extend_services_form = None
        self.fi = FilterInfo()
        self.raw = None
        self.saved_raw = None
        self.dock_area = DockArea(self)
        self.gridLayout.addWidget(self.dock_area)
        self.eeg_dock = QWidgetDock('eeg', self.dock_area)
        self.dock_area.addDock(self.eeg_dock)
        self.psd_dock = None
        self.tpm_dock = None
        self.select_signals_for_psd = None
        self.select_time_span_for_tpm = None
        self.export_form = None
        self.t_max = None  # 最后一次sample的开始时间
        self.groupbox_list = [self.groupBox_1, self.groupBox_2, self.groupBox_3, self.groupBox_4, self.groupBox_5]
        self.color_dialog = None
        self.ebm = EEGBrowserManager(self)
        self.setStyleSheet(ThemeColorConfig.get_ui_ss())
        self.init_ui()

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
        self.theme_btn.clicked.connect(self.change_theme)
        self.jmp_btn.clicked.connect(self.jump_to)
        self.fs_btn.clicked.connect(self.full_screen)
        self.export_btn.clicked.connect(self.export_edf)
        self.ea_btn.clicked.connect(self.extend_services)
        self.listWidget.itemClicked.connect(self.show_ann)
        self.listWidget.itemDoubleClicked.connect(self.jump_to_ann)
        self.rename_btn.clicked.connect(self.rename_des)
        self.delete_btn.clicked.connect(self.delete_des)
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
            text = 'Loaded: ' + adr.split('/')[-1]
            self.label.setToolTip(adr.split('/')[-1])
            if len(text) > 37:
                text = text[:34] + '...'
            self.label.setText(text)

            # self.change_size_cbx.setCurrentIndex(1)

            self.t_max = self.raw.times[-1]
            self.end_time.setMaximum(self.t_max)
            self.start_time.setMaximum(self.t_max)
            self.duration.setMaximum(self.t_max)

        self.plot_eeg(self.raw)
        self.get_ann()
        self.filter_cbx.setEnabled(True)

    def plot_eeg(self, raw, bg_color=None):
        """
        mne-qt-browser
        """
        self.ebm.create_eeg_browser(raw, bg_color)
        self.eeg_dock.set_widget(self.ebm.get_eeg_browser())
        # print(self.ebm.eeg_browser is self.eeg_dock.widget)
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
        self.plot_eeg(self.raw)
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
        if self.ebm.eeg_browser is None:
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
            color = PSDEnum.COLORS.value[i % len(PSDEnum.COLORS.value)]
            self.raw.compute_psd(fmin=para_list[1], fmax=para_list[2], picks=ch_name).plot(
                axes=axs[i], color=color, spatial_colors=False, dB=True, amplitude=False, show=False)
            axs[i].set_title(f'PSD for {ch_name}')
            for line in axs[i].lines:  # 遍历该轴中的所有线条
                line.set_path_effects([
                    Stroke(linewidth=2, foreground="gold", alpha=0.3),  # 外层光晕
                    Stroke(linewidth=2, foreground=color, alpha=0.5),  # 中层光晕
                    Normal()  # 正常线条
                ])
                line.set_linewidth(2)  # 设置主线条宽度
        tight_layout()  # 调整子图布局

        if self.psd_dock is not None:
            self.psd_dock.close()
            self.psd_dock = None

        psd_plot = FigureCanvas(fig)  # 绑定Figure到Canvas上
        psd_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        psd_plot.setFixedHeight(num_plots * 250)  # 每个子图固定高度

        self.psd_dock = SaveDock('psd', area=self.dock_area, canvas=psd_plot, hideTitle=False, enable_scroll=True)
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
        if self.ebm.eeg_browser is None:
            return

        self.select_time_span_for_tpm = SelectTimeSpanForTPM(self)
        self.select_time_span_for_tpm.show()
        self.diary.debug('MainForm —> SelectTimeSpanForTPM')

    def plot_tpm(self, span_list=None):
        """
        绘制topomap图
        """
        intersection = set(ChannelEnum.TPM.value) & set(self.eeg_list)
        if not intersection:
            QMessageBox.warning(self, 'Warning', 'The selected channel(s) is(are) not included in TPM channels!')
            return

        fig, axes = subplots(1, 5)
        raw = self.raw.copy().pick(picks=list(intersection))
        raw.set_montage(EdfUtil.get_montage())

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
        """
        custom_amp = self.custom_amp.value() if flag == "c" else None
        self.ebm.change_amplitude(flag, custom_amp)

    def change_window_size(self, custom):
        """
        改变窗口尺寸
        """
        if not custom:
            dur = int(self.change_size_cbx.currentText()[:-1])
        else:
            dur = self.custom_size.value()
        self.ebm.change_duration(dur)

    def change_num_signals(self, flag):
        """
        增加/减少window内channel数
        """
        self.ebm.change_num_signals(flag)

    def change_theme(self):
        """
        改变界面外观
        """
        if self.ebm.eeg_browser is None:
            return

        if not self.color_dialog:
            self.color_dialog = ColorDialog(QColor(0, 255, 255), "Choose Background Color for EEG", self,
                                            enableAlpha=True)
            self.color_dialog.colorChanged.connect(lambda color: self.plot_eeg(self.raw, color.name()))

            # theme = self.theme_cbx.currentText()
        # if theme == 'Light':
        #     app.setStyleSheet(load_stylesheet(qt_api='pyqt5', palette=LightPalette()))

        # 显示对话框
        self.color_dialog.exec()

        # elif theme == 'Dark':
        #     app.setStyleSheet(load_stylesheet_pyqt5())
        # elif theme == 'dark_teal':
        #     # qt-material
        #     apply_stylesheet(app, theme='dark_blue.xml')
        # elif theme == '123':
        #     toggleTheme()

    def jump_to(self):
        """
        EEG图跳转至指定时间
        """
        self.ebm.jump_to(self.jmp_time.value())

    def full_screen(self):
        """
        EEG图全屏
        """
        self.ebm.full_screen()

    def export_edf(self):
        """
        Load ExportForm.
        """
        if self.ebm.eeg_browser is None:
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
        if self.ebm.eeg_browser is None:
            return

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
            self.ebm.jump_to(self.raw.annotations[ann_idx]['onset'])

    def create_ann(self):
        """
        创建事件
        """
        if self.ann_txt.text() == '':
            return
        description = self.ann_txt.text()
        onset = self.start_time.value()
        duration = self.duration.value()
        self.ebm.create_ann(description, onset, duration)
        # self.raw.annotations.append(onset, duration, description)
        self.diary.info('Create Ann:\n'
                        f'onset: {onset}\n,'
                        f'duration: {duration}\n'
                        f'des: {description}\n'
                        f'Max time: {self.t_max}')
        self.get_ann()

    def rename_des(self):
        """
        更改事件名
        """
        ann_idx = self.listWidget.currentRow()
        if ann_idx >= 0:
            description = self.ann_txt.text()
            self.ebm.rename_des(ann_idx, description)
            self.diary.info(f'Rename Ann: {self.raw.annotations.description[ann_idx]} to {description}')
            self.get_ann()

    def delete_des(self):
        """
        删除相同事件名的全部事件
        """
        description = self.ann_txt.text()
        if description != '':
            self.ebm.delete_des(description)
            self.diary.info(f'Delete Ann: {description}')
            self.get_ann()

    def clear_ann(self):
        """
        清空事件
        """
        self.ebm.clear_ann()
        self.diary.info('Clear Ann')
        self.get_ann()

    def autoset_end(self):
        """
        控制时间
        """
        if self.ebm.eeg_browser is None:
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
        if self.ebm.eeg_browser is None:
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
                self.ebm.filter.exit_full_screen()

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


def set_global_mode(app):
    if ThemeColorConfig.theme == "light":
        app.setStyleSheet(load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    elif ThemeColorConfig.theme == "dark":
        style.use('dark_background')
        app.setStyleSheet(load_stylesheet_pyqt5())
        toggleTheme()
    rcParams['savefig.facecolor'] = ThemeColorConfig.get_eai_bg()

    # toggle_global_mode
    # if ThemeColorConfig.theme == "light":
    #     ThemeColorConfig.theme = 'dark'
    #     style.use('dark_background')
    #     app.setStyleSheet(load_stylesheet_pyqt5())
    #     toggleTheme()
    # elif ThemeColorConfig.theme == "dark":
    #     ThemeColorConfig.theme = 'light'
    #     style.use('default')
    #     app.setStyleSheet(load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    # self.setStyleSheet(ThemeColorConfig.get_ui_ss())
    # rcParams['savefig.facecolor'] = ThemeColorConfig.get_eai_bg()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(argv)
    app.setWindowIcon(QIcon(AddressConfig.get_icon_adr("icon", 'icon')))  # icon

    ThemeColorConfig.theme = "light"
    set_global_mode(app)

    main_form = MainForm()
    main_form.show()
    exit(app.exec_())
