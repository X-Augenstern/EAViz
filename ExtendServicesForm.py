from PyQt5.QtCore import QUrl, QPropertyAnimation, QEasingCurve
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QWidget, QTabBar, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor
from utils.config import IndexConfig, ModelConfig, AddressConfig
from math import floor
from collections import Counter
from utils.threads import (StatisticsThread, SeiDESAThread, ADThread, SDTemplateThread, SDSemanticsThread, HFOThread,
                           VDThread, VDModelThread)
from utils.custom_widgets import HoverLabel, MultiFuncEdit, SaveLabel
from utils.config import ChannelEnum,ThemeColorConfig
from cv2 import resize, cvtColor, COLOR_BGR2RGB
from torch import from_numpy
from AD.APSD import APSD
from HFO.hfo import show_plot
from ui.extend_services_form_cur import Ui_Form


# ui_extend_services_form = uic.loadUiType('ui/extend_services_form.ui')[0]

class ExtendServicesForm(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(ExtendServicesForm, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()
        self.setStyleSheet(ThemeColorConfig.get_ui_ss())
        if self.parent.adr is not None:
            text = self.parent.adr.split('/')[-1]
            self.adr_lbl.setToolTip(text)
            if len(text) > 37:
                text = text[:34] + '...'
            self.adr_lbl.setText(text)

        # ----------------------- UI -----------------------
        # 隐藏标签页
        self.tabBar = self.tabWidget.findChild(QTabBar)
        self.tabBar.hide()

        # 隐藏专用控件
        self.ad_unique_widgets()
        self.auto_annotate_widgets()

        # listWidget固定高度
        # self.listWidget.setFixedHeight(400)

        # ESC+SD
        self.seid_esa_fm = SaveLabel(self)
        self.seid_esa_feature = SaveLabel(self)
        self.seid_esa_res = SaveLabel(self)
        self.fm_layout.addWidget(self.seid_esa_fm)
        self.feature_layout.addWidget(self.seid_esa_feature)
        self.res_layout.addWidget(self.seid_esa_res)
        # AD
        self.ad_topo = SaveLabel(self)
        self.topo_layout.addWidget(self.ad_topo)
        self.ad_res = HoverLabel(self)
        self.ad_res_layout.addWidget(self.ad_res)
        # SpiD
        self.sd_res = HoverLabel(self)
        self.sd_res_layout.addWidget(self.sd_res)
        # ECharts
        self.ad_wev = QWebEngineView(self)
        self.sd_wev = QWebEngineView(self)
        self.ad_wev.page().setBackgroundColor(QColor(ThemeColorConfig.get_eai_bg()))
        self.sd_wev.page().setBackgroundColor(QColor(ThemeColorConfig.get_eai_bg()))
        self.ad_idx_layout.addWidget(self.ad_wev)
        self.sd_idx_layout.addWidget(self.sd_wev)
        # VD
        self.input_adr_le = MultiFuncEdit(self)
        self.output_adr_le = MultiFuncEdit(self, func=1)
        self.vd_gridLayout.addWidget(self.input_adr_le, 0, 3, 1, 1)  # row col rolSpan colSpan
        self.vd_gridLayout.addWidget(self.output_adr_le, 1, 3, 1, 1)

        # ----------------------- Attribute -----------------------
        self.raw = None
        self.check = False
        self.statistic_thread = None
        # 同步信号，防止在initialize方法中同步service_cbx和service_cbx1的过程中发生循环调用。
        # 如果去掉is_syncing标志，可能会导致循环调用问题：当一个组合框的索引更改时，会触发对另一个组合框的更改，而这又会反过来触发第一个组合框的更改，从而陷入无限循环。
        self.is_syncing = False
        # 防止span1、span2互相影响
        self.is_updating = False
        # ESC+SD
        self.seid_esa_mod = None
        self.seid_esa_thread = None
        # AD
        self.parse = None
        self.ad_thread = None
        # SpiD
        self.sd_tem_thread = None
        self.sd_sem_thread = None
        # SRD
        self.hfo_thread = None
        # VD
        self.vd_thread = None
        self.vd_mod_thread = None  # 标记为self才不会卡死进程！
        self.animi = None

    def init_ui(self):
        self.service_cbx.currentIndexChanged.connect(lambda: self.initialize(self.service_cbx.currentIndex()))
        self.service_cbx1.currentIndexChanged.connect(lambda: self.initialize(self.service_cbx1.currentIndex()))
        self.model_cbx.currentIndexChanged.connect(lambda: self.load_model(self.model_cbx.currentText()))
        self.tpm_btn.clicked.connect(lambda: self.repaint_tpm())
        if self.parent.raw is not None:
            self.change_signal_btn.clicked.connect(self.parent.change_signals)
        self.list_btn.clicked.connect(self.vd_setting_box)

    def initialize(self, index):
        """
        界面初始化
        """
        if self.is_syncing:
            return
        self.is_syncing = True
        self.service_cbx.setCurrentIndex(index)
        self.service_cbx1.setCurrentIndex(index)

        # 初始化
        self.to_dur_lbl.setText('to')  # SD
        self.span2.setDecimals(2)
        self.span2.setMinimum(0)
        self.tabWidget.setCurrentIndex(index)
        self.check = False
        self.model_cbx.clear()
        self.listWidget.clear()
        self.check_and_disconnect()
        self.ad_unique_widgets()
        self.auto_annotate_widgets()
        self.vd_hide_widgets(True)
        self.gb_2.setTitle('Channel for Statistics Calculating:')

        if index == IndexConfig.SeiD_ESA_ui_idx:
            self.add_model(ModelConfig.SeiD_ESA_model)
            self.set_process_txt(1, service_name='ESC+SD')
        elif index == IndexConfig.AD_ui_idx:
            self.add_model(ModelConfig.AD_model)
            self.set_process_txt(1, service_name='AD')
            self.ad_unique_widgets(True)
            self.auto_annotate_widgets(True)
        elif index == IndexConfig.SD_ui_idx:
            self.add_model(ModelConfig.SD_model)
            self.set_process_txt(1, service_name='SpiD')
            self.auto_annotate_widgets(True)
        elif index == IndexConfig.HFO_ui_idx:
            self.add_model(ModelConfig.HFO_model)
            self.set_process_txt(1, service_name='SRD')
            self.gb_2.setTitle('Channel for Analysing:')
        elif index == IndexConfig.VD_ui_idx:
            # self.add_model(ModelConfig.VD_model)
            self.set_process_txt(1, service_name='VD')
            self.vd_model_cbx.setToolTip(ModelConfig.get_des('VD'))
            self.vd_hide_widgets(False)
            if self.vd_thread is None:
                self.vd_ini()

        self.is_syncing = False

    def add_model(self, model_list):
        """
        加载模型进列表
        """
        for model in model_list:
            self.model_cbx.addItem(model)

    def load_model(self, model_name):
        """
        选择模型并检查当前raw
        """
        if model_name == '':
            self.model_cbx.setToolTip('')
            return
        self.model_cbx.setToolTip(ModelConfig.get_des(model_name))
        if model_name in ModelConfig.SeiD_ESA_model:
            self.seid_esa_mod = model_name
            if self.check is False:
                self.check_input([1000, 4.00, IndexConfig.SeiD_ESA_idx])
        elif model_name in ModelConfig.AD_model:
            self.parse = model_name.split('_')
            # 需要一个完整的文件路径来正确地定位文件
            self.ad_wev.load(QUrl.fromLocalFile(AddressConfig.get_ad_adr('idx', self.parse[0])))
            if self.check is False:  # 仅切换模型不用二次检查
                self.check_input([1000, 11.00, IndexConfig.AD_idx])
        elif model_name in ModelConfig.SD_model:
            self.sd_wev.load(QUrl.fromLocalFile(AddressConfig.get_sd_adr('idx')))
            self.show_local_img(self.sd_fam, AddressConfig.get_sd_adr('fam'))
            self.check_and_disconnect()
            self.listWidget.clear()
            if model_name == 'Template Matching':
                self.check_input([500, 0.3, IndexConfig.SD_idx])
            elif model_name == 'Unet+ResNet34':
                self.check_input([500, 30.00, IndexConfig.SD_idx])
        elif model_name == 'MKCNN':
            if self.check is False:
                self.check_input([1000, 1, IndexConfig.HFO_idx])

    def check_input(self, check_list):
        """
        检查raw是否满足检测模型要求[sfreq, span, group_idx]，若满足：
        1、自动按模型切片长度要求控制时间
        2、加载当前raw所含通道进列表
        3、可以计算各通道全局统计特征
        4、检测按钮启用
        """
        # Check .edf
        if self.parent.raw is None:
            self.set_process_txt(2)
            return
        else:
            self.raw = self.parent.raw.copy()

        # Check sfreq
        sfreq = self.raw.info['sfreq']
        if sfreq != check_list[0]:
            self.set_process_txt(3, sfreq=check_list[0])
            return

        # Check t_max
        t_max = floor(self.raw.n_times / sfreq * 100) / 100
        if t_max < check_list[1]:
            self.set_process_txt(4, min_time=check_list[1])
            return

        # Check chns
        chns = []
        if check_list[2] == IndexConfig.SeiD_ESA_idx:
            chns = ChannelEnum.CH21.value
        elif check_list[2] == IndexConfig.AD_idx or check_list[2] == IndexConfig.SD_idx:
            chns = ChannelEnum.CH19.value

        if chns and Counter(self.raw.ch_names) != Counter(chns):
            self.set_process_txt(5)
            return
        else:
            self.set_process_txt(6)

        # Prepare
        self.span1.setValue(0.00)
        self.span2.setValue(check_list[1])
        self.span2.setMinimum(check_list[1])
        self.listWidget.addItems(self.raw.ch_names)
        self.span1.valueChanged.connect(lambda: self.span_control(1, check_list[1], t_max))
        self.span2.valueChanged.connect(lambda: self.span_control(2, check_list[1]))

        if check_list[2] != IndexConfig.HFO_idx:
            self.listWidget.itemClicked.connect(lambda: self.run_calcul_sta(g_or_r='g', group_idx=check_list[2]))

        if check_list[2] == IndexConfig.SeiD_ESA_idx:
            self.start_btn.clicked.connect(lambda: self.run_seid_esa(self.seid_esa_mod))
        if check_list[2] == IndexConfig.AD_idx:
            self.start_btn.clicked.connect(lambda: self.run_ad(self.parse))  # 会多次绑定，导致按一下start就会执行绑定次数的槽函数
        if check_list[2] == IndexConfig.SD_idx:
            self.span1.valueChanged.disconnect()
            self.span2.valueChanged.disconnect()
            if check_list[1] == 0.3:
                self.to_dur_lbl.setText('to')
                self.span2.setDecimals(2)
                self.span2.setValue(check_list[1])
                self.span1.valueChanged.connect(lambda: self.min_span_control(0, check_list[1], t_max))
                self.span2.valueChanged.connect(lambda: self.min_span_control(1, check_list[1], t_max))
                self.start_btn.clicked.connect(self.run_sd_template)
            elif check_list[1] == 30:
                self.to_dur_lbl.setText('Dur(s): 30s *')
                self.span2.setDecimals(0)
                self.span2.setMinimum(1)
                self.span2.setValue(1)
                self.span1.valueChanged.connect(lambda: self.multi_span_control(0, check_list[1], t_max))
                self.span2.valueChanged.connect(lambda: self.multi_span_control(1, check_list[1], t_max))
                self.start_btn.clicked.connect(self.run_sd_semantics)
        if check_list[2] == IndexConfig.HFO_idx:
            self.span1.valueChanged.disconnect()
            self.span2.valueChanged.disconnect()
            self.span2.setDecimals(2)
            self.span2.setValue(check_list[1])
            self.span1.valueChanged.connect(lambda: self.min_span_control(0, check_list[1], t_max))
            self.span2.valueChanged.connect(lambda: self.min_span_control(1, check_list[1], t_max))
            # 在使用 lambda 时包含所有必要的调用括号。如果 self.run_hfo 需要接收特定的参数，lambda 还可以用来提供这些参数
            self.start_btn.clicked.connect(self.run_hfo)
        self.check = True
        self.parent.diary.info('Pass Checking')

    def span_control(self, index, span, t_max=None):
        """
        自动按模型切片长度控制时间 span
        """
        if self.is_updating:
            return  # 如果正在更新，直接返回，防止循环更新
        self.is_updating = True

        if index == 1:
            if self.span1.value() > t_max - span:
                self.span2.setValue(t_max)
                self.span1.setValue(t_max - span)
            else:
                self.span2.setValue(self.span1.value() + span)
        if index == 2:
            if self.span2.value() - span < 0:
                self.span2.setValue(span)
                self.span1.setValue(0)
            else:
                self.span1.setValue(self.span2.value() - span)

        self.is_updating = False  # 更新完成，解除标志

    def min_span_control(self, index, min_span, t_max):
        """
        自动按模型切片长度控制时间 >= min_span
        """
        self.span1.setValue(min(self.span1.value(), t_max - min_span))
        self.span2.setValue(min(self.span2.value(), t_max))

        if self.span2.value() - self.span1.value() < min_span:
            if index == 0:
                new_span2 = min(self.span1.value() + min_span, t_max)
                self.span2.setValue(new_span2)
            elif index == 1:
                new_span1 = max(self.span2.value() - min_span, 0)
                self.span1.setValue(new_span1)

    def multi_span_control(self, index, span, t_max):
        """
        自动按模型切片长度要求控制时间 multiple of span
        """
        if index == 0:
            self.span1.setValue(min(self.span1.value(), t_max - span))

        if self.span1.value() + self.span2.value() * span > t_max:
            self.span2.setValue(floor((t_max - self.span1.value()) / span))

    def run_calcul_sta(self, g_or_r, group_idx, freq=None, start=None, stop=None):
        """
        计算统计特征
        """
        selected_indexed = self.listWidget.selectedIndexes()
        if not selected_indexed:
            return
        ch_idx = selected_indexed[0].row()  # 获取选中通道索引

        self.statistic_thread = StatisticsThread(g_or_r, group_idx, ch_idx, self.raw.copy(), freq, start, stop)
        self.statistic_thread.res_signal.connect(self.toggle_lbl_to_show)
        self.statistic_thread.start()

        self.parent.diary.info(f'calculate_statistics {g_or_r} + {group_idx}')

    def toggle_lbl_to_show(self, group, g_r, data_list):
        """
        切换显示统计特征的QLabel
        """
        lbl_dict = {
            (IndexConfig.SeiD_ESA_idx, 'r'): [[self.r_mean, self.r_std, self.r_var],
                                              [self.r_d, self.r_t, self.r_a, self.r_b, self.r_g]],
            (IndexConfig.SeiD_ESA_idx, 'g'): [[self.g_mean, self.g_std, self.g_var],
                                              [self.g_d, self.g_t, self.g_a, self.g_b, self.g_g]],
            (IndexConfig.AD_idx, 'r'): [[self.r_mean_1, self.r_std_1, self.r_var_1],
                                        [self.r_d_1, self.r_t_1, self.r_a_1, self.r_b_1, self.r_g_1]],
            (IndexConfig.AD_idx, 'g'): [[self.g_mean_1, self.g_std_1, self.g_var_1],
                                        [self.g_d_1, self.g_t_1, self.g_a_1, self.g_b_1, self.g_g_1]],
            (IndexConfig.SD_idx, 'r'): [[self.r_mean_2, self.r_std_2, self.r_var_2],
                                        [self.r_d_2, self.r_t_2, self.r_a_2, self.r_b_2, self.r_g_2]],
            (IndexConfig.SD_idx, 'g'): [[self.g_mean_2, self.g_std_2, self.g_var_2],
                                        [self.g_d_2, self.g_t_2, self.g_a_2, self.g_b_2, self.g_g_2]],
            # (IndexConfig.HFO_idx, 'r'): [[self.r_mean_3, self.r_std_3, self.r_var_3],
            #                              [self.r_d_3, self.r_t_3, self.r_a_3, self.r_b_3, self.r_g_3]],
            # (IndexConfig.HFO_idx, 'g'): [[self.g_mean_3, self.g_std_3, self.g_var_3],
            #                              [self.g_d_3, self.g_t_3, self.g_a_3, self.g_b_3, self.g_g_3]]
        }
        label_list = lbl_dict.get((group, g_r), [])  # 如果字典中没有匹配项，则返回一个空列表
        if label_list is not []:
            self.show_statistics(label_list, data_list)

    @staticmethod
    def show_statistics(label_list, data_list):
        """
        大批量QLabel设置文本
        label_list: list[list]
        data_list: list[list]
        """
        for labels, datum in zip(label_list, data_list):
            for label, data in zip(labels, datum):
                label.setText(f'{data}')

    def run_seid_esa(self, model):
        """
        SeiD/ESA 线程启动
        """
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.SeiD_ESA_idx, freq=1000, start=start_time, stop=stop_time)

        self.parent.diary.info('使用ESC+SD线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'model:{model}')

        self.seid_esa_thread = SeiDESAThread(self.raw.copy(), start_time, model)
        if model == 'DSMN-ESS':
            self.seid_esa_thread.fm_signal.connect(self.seid_esa_fm.update_img)
            self.seid_esa_thread.feature_signal.connect(self.seid_esa_feature.update_img)
            self.seid_esa_thread.res_signal.connect(self.seid_esa_res.update_img)
        elif model == 'R3DClassifier':
            self.seid_esa_thread.fm_signal.connect(self.seid_esa_fm.update_img)
            self.seid_esa_thread.feature_signal.connect(self.seid_esa_feature.update_img)
            self.seid_esa_thread.res_signal.connect(self.seid_esa_res.update_img)
        self.seid_esa_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name=model))
        self.seid_esa_thread.start()

    def ad_unique_widgets(self, show=False):
        """
        AD 独有控件
        """
        self.ad_lbl_1.setVisible(show)
        self.ad_lbl_2.setVisible(show)
        self.n_btn.setVisible(show)
        self.eb_btn.setVisible(show)
        self.fe_btn.setVisible(show)
        self.ce_btn.setVisible(show)
        self.te_btn.setVisible(show)
        self.u_btn.setVisible(show)
        self.tpm_cbx.setVisible(show)
        self.tpm_btn.setVisible(show)

    def auto_annotate_widgets(self, show=False):
        """
        自动添加检测结果控件
        """
        self.ann_lbl.setVisible(show)
        self.ann_cbx.setVisible(show)

    def run_ad(self, model):
        """
        AD 线程启动
        """
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.AD_idx, freq=1000, start=start_time, stop=stop_time)

        arti_list = self.get_arti_list()
        fb_idx = self.tpm_cbx.currentIndex()
        auto = self.ann_cbx.isChecked()
        self.parent.diary.info('使用AD线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'model1: {model[0]}\n'
                               f'model2: {model[1]}\n'
                               f'artifact list: {arti_list}\n'
                               f'tpm:{fb_idx}\n'
                               f'annotate: {auto}')

        self.ad_thread = ADThread(self.raw.copy(), int(start_time), int(stop_time), model[0], model[1], arti_list,
                                  fb_idx, auto)
        # self.ad_thread.topo_signal.connect(lambda: self.show_local_img(self.ad_topo, AddressConfig.get_ad_adr('topo')))
        # self.ad_thread.res_signal.connect(lambda: self.show_local_img(self.ad_res, AddressConfig.get_ad_adr('res')))
        self.ad_thread.topo_signal.connect(self.ad_topo.update_img)
        self.ad_thread.res_signal.connect(lambda x: self.ad_res.update_img(img_data=x))
        self.ad_thread.ann_signal.connect(self.annotate_result_overlap)
        self.ad_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name=f'{model[0]}+{model[1]}'))
        self.ad_thread.finish_signal.connect(
            lambda: self.tpm_btn.setEnabled(True) if not self.tpm_btn.isEnabled() else None)
        self.ad_thread.start()

    def get_arti_list(self):
        """
        获取伪迹选中列表
        """
        arti = []
        if self.n_btn.isChecked():
            arti.append(0)
        if self.eb_btn.isChecked():
            arti.append(1)
        if self.fe_btn.isChecked():
            arti.append(2)
        if self.ce_btn.isChecked():
            arti.append(3)
        if self.te_btn.isChecked():
            arti.append(4)
        if self.u_btn.isChecked():
            arti.append(5)
        return arti

    def repaint_tpm(self):
        """
        绘制伪迹topomap图
        """
        if self.raw is None:
            return
        # 滤波
        raw1 = self.raw.copy().notch_filter(freqs=50)
        raw_filtered = raw1.filter(l_freq=1, h_freq=70)

        raw_data = raw_filtered.get_data()  # (19,1200000) 1200s

        n_data = from_numpy(raw_data)
        fb_idx = self.tpm_cbx.currentIndex()
        self.ad_topo.update_img(
            APSD(n_data, self.span1.value(), self.span2.value(), fb_idx)
        )
        self.parent.diary.info(f'tpm:{fb_idx}')

    def run_sd_template(self):
        """
        SD 模板匹配线程启动
        """
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.SD_idx, freq=500, start=start_time, stop=stop_time)

        auto = self.ann_cbx.isChecked()
        self.parent.diary.info('使用模板匹配线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'stop_time: {stop_time}\n'
                               f'annotate: {auto}')

        self.sd_tem_thread = SDTemplateThread(self.raw.copy(), start_time, stop_time, auto)
        self.sd_tem_thread.res_signal.connect(lambda x, y: self.sd_res.update_img(img_data=x, raw=y))
        self.sd_tem_thread.ann_signal.connect(self.annotate_result_overlap)
        self.sd_tem_thread.swi_signal.connect(lambda x: self.sd_swi.setText('SWI(Spike wave index):' + x))
        self.sd_tem_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name='Template Matching'))
        self.sd_tem_thread.start()

    def run_sd_semantics(self):
        """
        SD 语义分割线程启动
        """
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = start_time + self.span2.value() * 30
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.SD_idx, freq=500, start=start_time, stop=stop_time)

        auto = self.ann_cbx.isChecked()
        self.parent.diary.info('使用语义分割线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'stop_time: {stop_time}\n'
                               f'annotate: {auto}')

        self.sd_sem_thread = SDSemanticsThread(self.raw.copy(), start_time, stop_time, auto)
        self.sd_sem_thread.res_signal.connect(lambda x, y: self.sd_res.update_img(img_data=x, raw=y))
        self.sd_sem_thread.ann_signal.connect(self.annotate_result_overlap)
        self.sd_sem_thread.swi_signal.connect(lambda x: self.sd_swi.setText('SWI(Spike wave index):' + x))
        self.sd_sem_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name='Semantic Segmentation'))
        self.sd_sem_thread.start()

    def annotate_result_overlap(self, ann_list, des_list, start_time, end_time):
        """
        添加检测结果：（防止重复添加相同事件，且若事件前后有重叠，合并两个重叠事件为一个，直到与后续事件没有重叠为止）
        ann_list: [[onset, duration, description], [...]]
        test: AD: 1000data_ad.edf 0-11s 1-12s 2-13s
              SD: test_19_sd.edf 0-10s 1-11s 2-12s 29-39s 30-40s
        """
        if len(ann_list) > 0:
            # ann = self.parent.raw.annotations 使用 ann.crop() 依然会影响到 self.parent.raw.annotations 需要使用copy
            ann = self.parent.raw.annotations.copy()
            # 切片前的事件数 < 而不能 <=，刚好 = 也会记作一次事件
            before_crop = sum(1 for onset, dur in zip(ann.onset, ann.duration) if onset + dur < start_time)
            # 切片
            end_time = end_time - 1 / self.parent.raw.info['sfreq']
            ann.crop(tmin=start_time, tmax=end_time, use_orig_time=False)
            # 是否有事件在切片内
            des_in_ann = set(des for des in des_list if des in ann.description)
            if len(des_in_ann) > 0:
                # 切片内包含的事件
                ann_exist = [[onset, dur, des] for onset, dur, des in zip(ann.onset, ann.duration, ann.description)]
                # 切片内包含的在des_in_ann中的事件，并去除dur为0的事件
                ann_exist_des = [a for a in ann_exist if a[2] in des_in_ann and a[1] != 0]
                # ann_list.remove(a for a in ann_list if a in ann_exist)  # list.remove() 期望传递要删除的具体元素值，而不是通过条件来筛选
                if len(ann_exist_des) > 0:
                    # 用于循环的ann_list
                    temp_ann_list = ann_list.copy()
                    # 旧事件索引
                    delete_list = []
                    # 用ann_list去匹配ann_exist_des，当b在与前一个a重叠处理后，又可能与下一个a重叠，用递归将两/多个事件的重叠连接起来
                    self.parent.diary.info('ann_exist_des:\n'
                                           f'{ann_exist_des}\n'
                                           'temp_ann_list:\n'
                                           f'{temp_ann_list}')
                    for b in temp_ann_list:
                        cur_b = b

                        def overlap(new_ann, new_b):
                            for i, a in enumerate(new_ann):
                                if new_b[2] != a[2]:  # 若事件不相同
                                    continue
                                if new_b[0] + new_b[1] < a[0]:  # 不重叠
                                    ann_list.append(new_b)  # 保存新b
                                    ann_list.remove(cur_b)  # 删除旧b
                                    return True
                                # 内部重合
                                if new_b[0] == a[0] > start_time and new_b[0] + new_b[1] == a[0] + a[1] < end_time:
                                    ann_list.remove(cur_b)
                                    return True
                                # 重叠+边缘重合
                                if (new_b[0] + new_b[1] >= a[0] and new_b[0] <= a[0] + a[1]) \
                                        or new_b[0] == a[0] == start_time \
                                        or new_b[0] + new_b[1] == a[0] + a[1] == end_time:
                                    # 旧事件索引
                                    delete_idx = ann_exist.index(a) + before_crop
                                    delete_list.append(delete_idx)
                                    new_ann.remove(a)
                                    # a其实都是检测结果，应当看作用当前的检测结果去匹配先前的检测结果
                                    ann_exist_des.remove(a)  # a、b都是从前到后匹配，重叠的a就直接删掉，避免a的重用并加速
                                    a_start = self.parent.raw.annotations.onset[delete_idx]  # raw中的起始时间
                                    a_end = a_start + self.parent.raw.annotations.duration[delete_idx]  # raw中的结束时间
                                    onset = min(new_b[0], a_start)  # 新事件起始时间也要用在raw中的起始时间作比较
                                    # 新事件的dur应该是max[onset+dur]-min[onset] （注：未考虑到若new_ann中有相同事件）
                                    new_b = [onset, max(new_b[0] + new_b[1], a_end) - onset, new_b[2]]
                                    if len(new_ann) > 0:  # 可继续递归
                                        # 递归，设立标志位，一旦递归函数返回True就结束外部循环
                                        if overlap(new_ann, new_b):
                                            return
                                    else:
                                        ann_list.append(new_b)
                                        ann_list.remove(cur_b)
                                        return True
                                if i == len(new_ann) - 1:  # 已经遍历完了 new_b[0] > a[0] + a[1]
                                    ann_list.append(new_b)
                                    ann_list.remove(cur_b)
                                    return True

                        overlap(ann_exist_des.copy(), cur_b)
                    # 删除原先的事件（事件要最后删，否则会影响到旧事件索引值）
                    if len(delete_list) > 0:
                        delete_list.sort(reverse=True)
                        for idx in delete_list:
                            self.parent.raw.annotations.delete(idx)

        if len(ann_list) > 0:
            for a in ann_list:
                self.parent.raw.annotations.append(a[0], a[1], a[2])
            self.parent.plot_eeg(self.parent.raw, start=start_time)
            self.parent.get_ann()
            self.raw = self.parent.raw.copy()
        self.set_process_txt(10)

    def run_hfo(self):
        """
        HFO 线程启动
        """
        if self.model_cbx.currentText() == '':
            return
        selected_indexed = self.listWidget.selectedIndexes()
        if not selected_indexed:
            self.set_process_txt(14)
            return
        ch_idx = selected_indexed[0].row()  # 获取选中通道索引
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        # self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.HFO_idx, freq=1000, start=start_time, stop=stop_time)

        self.parent.diary.info('使用SRD线程进行检测，接口传入参数：\n'
                               f'ch_idx: {ch_idx}\n'
                               f'start time: {start_time}\n'
                               f'stop_time: {stop_time}\n')

        self.hfo_thread = HFOThread(self.raw.copy(), ch_idx, start_time, stop_time)
        self.hfo_thread.res_signal.connect(lambda x: self.plot_hfo(x, start_time))
        # self.hfo_thread.res_signal.connect(lambda x: x.plot())  # 此处已经把HFO事件加入到单通道的raw中了（绘制出来就有显示）
        self.hfo_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name='SRD'))
        self.hfo_thread.start()

    def plot_hfo(self, merged_raw, start_time):
        while self.hfo_layout.count():
            item = self.hfo_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.hfo_layout.addWidget(show_plot(merged_raw, start_time))

    def vd_ini(self):
        """
        VD 初始化线程
        """
        self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('start')))
        self.vd_stop_btn.setIcon(QIcon(AddressConfig.get_icon_adr('stop')))
        self.list_btn.setIcon(QIcon(AddressConfig.get_icon_adr('list')))
        self.vd_thread = VDThread()
        self.vd_thread.img_signal.connect(lambda x: self.vd_show_video_img(x, self.output_lbl))
        self.vd_thread.res_signal.connect(self.vd_res.addItem)
        self.vd_thread.percent_signal.connect(lambda x: self.progressBar.setValue(x))
        self.vd_thread.normal_finish_signal.connect(self.vd_stop)
        self.vd_thread.normal_finish_signal.connect(lambda: self.set_process_txt(8, model_name='VD'))
        self.vd_thread.abnormal_finish_signal.connect(lambda: self.set_process_txt(11))

        self.vd_mod_thread = VDModelThread(self.vd_thread.cfg)
        self.vd_mod_thread.mod_signal.connect(self.vd_thread.update_mod)
        self.vd_mod_thread.finish_signal.connect(lambda: self.set_process_txt(12))
        self.vd_mod_thread.finish_signal.connect(self.vd_activate)
        self.vd_mod_thread.start()

    def vd_hide_widgets(self, show=True):
        """
        VD 隐藏控件
        """
        self.gb_1.setVisible(show)
        self.gb_2.setVisible(show)
        self.gb_3.setVisible(show)

    def vd_setting_box(self):
        """
        VD 列表
        """
        width = self.setting_box.width()
        if width == 0:
            width_extend = 350
        else:
            width_extend = 0
        self.animi = QPropertyAnimation(self.setting_box, b"minimumWidth")
        self.animi.setDuration(500)
        self.animi.setStartValue(width)
        self.animi.setEndValue(width_extend)
        self.animi.setEasingCurve(QEasingCurve.InOutQuart)
        self.animi.start()

    def vd_activate(self):
        """
        VD 使能
        """
        self.vd_export_btn.clicked.connect(self.vd_export_res)
        # self.SISO.setEnabled(True)
        # self.MIMO.setEnabled(True)
        self.input_adr_le.setEnabled(True)
        self.output_adr_le.setEnabled(True)
        self.input_adr_le.textChanged.connect(self.vd_path_setting)
        self.output_adr_le.textChanged.connect(self.vd_path_setting)
        self.vd_save_cbx.setEnabled(True)
        self.iou_sb.setEnabled(True)
        self.iou_sd.setEnabled(True)
        self.conf_sb.setEnabled(True)
        self.conf_sd.setEnabled(True)
        # self.SISO.clicked.connect(self.vd_toggle)
        # self.MIMO.clicked.connect(self.vd_toggle)
        self.vd_save_cbx.clicked.connect(self.vd_save_setting)
        self.iou_sb.valueChanged.connect(lambda x: self.vd_change_value(x, 'iou_sb'))
        self.iou_sd.valueChanged.connect(lambda x: self.vd_change_value(x, 'iou_sd'))
        self.conf_sb.valueChanged.connect(lambda x: self.vd_change_value(x, 'conf_sb'))
        self.conf_sd.valueChanged.connect(lambda x: self.vd_change_value(x, 'conf_sd'))
        self.vd_start_btn.clicked.connect(self.vd_run_or_continue)
        self.vd_stop_btn.clicked.connect(self.vd_stop)

    def vd_path_setting(self):
        """
        VD 地址改变
        """
        self.vd_stop()
        input_list = self.input_adr_le.input_list
        if len(input_list) > 0:
            self.vd_thread.update_adr(input_list, self.output_adr_le.text())

    def vd_toggle(self):
        """
        VD 输入输出模式改变
        """
        if self.SISO.isChecked():
            self.input_adr_le.set_mode_and_ini(0)
            self.output_adr_le.set_mode_and_ini(0)
            self.vd_thread.s_or_m = 0
        if self.MIMO.isChecked():
            self.input_adr_le.set_mode_and_ini(1)
            self.output_adr_le.set_mode_and_ini(1)
            self.vd_thread.s_or_m = 1

    def vd_save_setting(self):
        """
        VD 保存
        """
        if self.vd_save_cbx.isChecked():
            self.vd_thread.save = True
        else:
            self.vd_thread.save = False

    def vd_change_value(self, x, flag):
        """
        VD 改变参数
        """
        if flag == 'iou_sb':
            self.iou_sd.setValue(int(x * 100))
        elif flag == 'conf_sb':
            self.conf_sd.setValue(int(x * 100))

        if flag == 'iou_sd':
            self.iou_sb.setValue(x / 100)
            self.vd_thread.iou_thres = x / 100
        elif flag == 'conf_sd':
            self.conf_sb.setValue(x / 100)
            self.vd_thread.conf_thres = x / 100

    def vd_run_or_continue(self):
        """
        VD 开始/暂停
        """
        if self.input_adr_le.text() == '':
            QMessageBox.warning(self, 'Warning', 'Please select the input file(s)!')
            return
        self.vd_thread.jump_out = False
        # 是否正在运行 不在运行返回true
        if self.vd_start_btn.isChecked():
            self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('pause')))
            self.vd_thread.is_continue = True
            self.set_process_txt(7)
            # 如果该线程还没开始则start
            if not self.vd_thread.isRunning():
                self.vd_res.clear()
                self.parent.diary.info('使用VD线程进行检测，接口传入参数：\n'
                                       f'input list: {self.vd_thread.input_list}\n'
                                       f'output address: {self.vd_thread.output_adr}\n'
                                       f'save: {self.vd_thread.save}')
                self.vd_thread.start()
        else:
            self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('start')))
            self.set_process_txt(13)
            self.vd_thread.is_continue = False

    def vd_stop(self):
        """
        VD 终止
        """
        self.vd_start_btn.setChecked(False)
        self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('start')))
        self.vd_thread.jump_out = True

    @staticmethod
    def vd_show_video_img(img, label):
        """
        VD 视频流
        """
        try:
            width = img.shape[1]
            height = img.shape[0]
            wid = label.width()
            hei = label.height()
            if width / height >= wid / hei:
                show = resize(img, (wid, int(height * wid / width)))
            else:
                show = resize(img, (int(width * hei / height), hei))
            show = cvtColor(show, COLOR_BGR2RGB)
            img = QImage(show.data, show.shape[1], show.shape[0], show.shape[2] * show.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(repr(e))

    def vd_export_res(self):
        """
        VD 导出结果
        """
        adr, _ = QFileDialog.getSaveFileName(self, 'Save .txt', '.', 'TXT files (*.txt)')
        if adr is None or len(adr) == 0:
            return
        else:
            with open(adr, 'w', encoding='utf-8') as file:
                for i in range(self.vd_res.count()):
                    item = self.vd_res.item(i)
                    file.write(item.text() + '\n')
            self.parent.diary.info('Save VD Results')

    def check_and_disconnect(self):
        """
        取消共用控件对槽函数的绑定
        """
        # start_btn
        num_receivers1 = self.start_btn.receivers(self.start_btn.clicked)  # 获取与按钮信号相关联的槽函数数量
        if num_receivers1 > 0:
            self.start_btn.clicked.disconnect()
            # self.start_btn.clicked.disconnect(self.run_offline_plot)  # 取消与指定槽函数的绑定（注意.clicked不要忘记！）

        # span1、span2
        num_receivers2 = self.span1.receivers(self.span1.valueChanged)
        if num_receivers2 > 0:
            self.span1.valueChanged.disconnect()

        num_receivers3 = self.span2.receivers(self.span2.valueChanged)
        if num_receivers3 > 0:
            self.span2.valueChanged.disconnect()

        # listWidget
        num_receivers4 = self.listWidget.receivers(self.listWidget.itemClicked)
        if num_receivers4 > 0:
            self.listWidget.itemClicked.disconnect()

    def show_local_img(self, label, img_path):
        """
        从本地加载图片
        """
        img = QImage()
        if img.load(img_path):
            pixmap = QPixmap.fromImage(img)
            label.setPixmap(pixmap)
        else:
            self.parent.diary.warning('Unable to load image.')

    def set_process_txt(self, status_code, service_name=None, sfreq=None, min_time=None, model_name=None):
        """
        :param status_code:
            1: service_name initialized!
            2: load .edf
            3: sfreq
            4: min_time
            5: chns does not meet
            6: chs met
            7: analysis in progress
            8: model_name analysis complete
            9: add analysis results
            10: adding complete
            11: analysis terminated
            12: VD model loaded
            13: analysis paused
            14: no chn is selected
        :param service_name: with 1
        :param sfreq: with 3
        :param min_time: with 4
        :param model_name: with 8
        """
        txt = {
            1: f'{service_name} successfully initialized!',
            2: 'Please load the .edf file!',
            3: f'Required sampling frequency {sfreq}Hz!',
            4: f'Required sampling time >= {min_time}s!',
            5: 'The current channel does not meet the requirements!',
            6: 'Channel requirements met. You can start analyse...',
            7: 'Analysis in progress, please wait...',
            8: f'{model_name} analysis complete!',
            9: 'Start annotating analysis results...',
            10: 'Annotating complete.',
            11: 'Analysis terminated.',
            12: 'VD model loaded!',
            13: 'Analysis paused.',
            14: 'Please select a channel for SRD analysing.'
        }

        diary_txt = {
            2: '请加载.edf文件！',
            3: f'要求.edf的采样频率为{sfreq}Hz！',
            4: f'要求.edf的时间 >= {min_time}s！',
            5: '当前通道不符合要求！',
            6: '当前通道符合要求，可以开始检测...',
            8: f'{model_name}检测完成！',
            9: '开始添加检测结果...',
            10: '检测结果添加完成！',
            11: 'VD检测终止！',
            14: '未选择用于SRD检测的通道'
        }

        if status_code in txt:
            self.process_text_lbl.setText(txt[status_code])
            if status_code in {7, 9}:
                self.process_text_lbl.repaint()
        if status_code in diary_txt:
            self.parent.diary.info(diary_txt[status_code])

    def closeEvent(self, event):
        """
        退出事件处理函数
        """
        self.parent.extend_services_form = None
        event.accept()  # 接受关闭事件
        self.parent.diary.debug('ExtendServicesForm —> MainForm')
