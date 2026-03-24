"""
基于引擎的扩展服务界面：完全基于原 ExtendServicesForm.py 构建，仅做以下改动：
1. 替换线程导入：utils.threads → engine.scheduler.offline_scheduler
2. 替换线程类名：ESCSDThread → ESCSDEngineThread 等
3. 其他逻辑（UI绑定、信号连接、参数传递）完全保持原样

这样可以确保：
- 原有界面的每一个 UI 交互行为完全不改变
- 算法逻辑通过引擎标准化接口执行
- 新增/替换模型只需要改 engine/ 目录，无需动 UI 代码
"""
from collections import Counter
from cv2 import resize, cvtColor, COLOR_BGR2RGB
from math import floor
from PyQt5.QtCore import QUrl, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QWidget, QTabBar, QMessageBox, QFileDialog
from torch import from_numpy

from AD.APSD import APSD
from engine.scheduler.offline_scheduler import (
    ESCSDEngineThread,
    ADEngineThread,
    SpiDEngineThread,
    SRDEngineThread,
    VDEngineThread,
)
from SRD.hfo import show_plot
from ui.extend_services_form_cur import Ui_Form
from utils.config import IndexConfig, ModelConfig, AddressConfig, ChannelEnum, ThemeColorConfig
from utils.custom_widgets import HoverLabel, MultiFuncEdit, SaveLabel
from utils.threads import VDModelThread


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

        # ESC_SD
        self.esc_sd_fm = SaveLabel(self)
        self.esc_sd_feature = SaveLabel(self)
        self.esc_sd_res = SaveLabel(self)
        self.fm_layout.addWidget(self.esc_sd_fm)
        self.feature_layout.addWidget(self.esc_sd_feature)
        self.res_layout.addWidget(self.esc_sd_res)
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
        self.is_syncing = False
        self.is_updating = False
        # ESC_SD
        self.esc_sd_mod = None
        self.esc_sd_thread = None
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
        self.vd_mod_thread = None
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
        if self.is_syncing:
            return
        self.is_syncing = True
        self.service_cbx.setCurrentIndex(index)
        self.service_cbx1.setCurrentIndex(index)

        # 初始化
        self.to_dur_lbl.setText('to')
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

        if index == IndexConfig.ESC_SD_ui_idx:
            self.add_model(ModelConfig.ESC_SD_model)
            self.set_process_txt(1, service_name='ESC+SD')
        elif index == IndexConfig.AD_ui_idx:
            self.add_model(ModelConfig.AD_model)
            self.set_process_txt(1, service_name='AD')
            self.ad_unique_widgets(True)
            self.auto_annotate_widgets(True)
        elif index == IndexConfig.SpiD_ui_idx:
            self.add_model(ModelConfig.SpiD_model)
            self.set_process_txt(1, service_name='SpiD')
            self.auto_annotate_widgets(True)
        elif index == IndexConfig.SRD_ui_idx:
            self.add_model(ModelConfig.SRD_model)
            self.set_process_txt(1, service_name='SRD')
            self.gb_2.setTitle('Channel for Analysing:')
        elif index == IndexConfig.VD_ui_idx:
            self.set_process_txt(1, service_name='VD')
            self.vd_model_cbx.setToolTip(ModelConfig.get_des('VD'))
            self.vd_hide_widgets(False)
            if self.vd_thread is None:
                self.vd_ini()

        self.is_syncing = False

    def add_model(self, model_list):
        self.model_cbx.addItems(model_list)

    def load_model(self, model_name):
        if model_name == '':
            self.model_cbx.setToolTip('')
            return
        self.model_cbx.setToolTip(ModelConfig.get_des(model_name))
        if model_name in ModelConfig.ESC_SD_model:
            self.esc_sd_mod = model_name
            if self.check is False:
                self.check_input([1000, 4.00, IndexConfig.ESC_SD_idx])
        elif model_name in ModelConfig.AD_model:
            self.parse = model_name.split('_')
            self.ad_wev.load(QUrl.fromLocalFile(AddressConfig.get_ad_adr('idx', self.parse[0])))
            if self.check is False:
                self.check_input([1000, 11.00, IndexConfig.AD_idx])
        elif model_name in ModelConfig.SpiD_model:
            self.sd_wev.load(QUrl.fromLocalFile(AddressConfig.get_spid_adr('idx')))
            self.show_local_img(self.sd_fam, AddressConfig.get_spid_adr('fam'))
            self.check_and_disconnect()
            self.listWidget.clear()
            if model_name == 'Template Matching':
                self.check_input([500, 0.3, IndexConfig.SpiD_idx])
            elif model_name == 'Unet+ResNet34':
                self.check_input([500, 30.00, IndexConfig.SpiD_idx])
        elif model_name == 'MKCNN':
            if self.check is False:
                self.check_input([1000, 1, IndexConfig.SRD_idx])

    def check_input(self, check_list):
        if self.parent.raw is None:
            self.set_process_txt(2)
            return
        else:
            self.raw = self.parent.raw.copy()

        sfreq = self.raw.info['sfreq']
        if sfreq != check_list[0]:
            self.set_process_txt(3, sfreq=check_list[0])
            return

        t_max = floor(self.raw.n_times / sfreq * 100) / 100
        if t_max < check_list[1]:
            self.set_process_txt(4, min_time=check_list[1])
            return

        chns = []
        if check_list[2] == IndexConfig.ESC_SD_idx:
            chns = ChannelEnum.CH21.value
        elif check_list[2] == IndexConfig.AD_idx or check_list[2] == IndexConfig.SpiD_idx:
            chns = ChannelEnum.CH19.value

        if chns and Counter(self.raw.ch_names) != Counter(chns):
            self.set_process_txt(5)
            return
        else:
            self.set_process_txt(6)

        self.span1.setValue(0.00)
        self.span2.setValue(check_list[1])
        self.span2.setMinimum(check_list[1])
        self.listWidget.addItems(self.raw.ch_names)
        self.span1.valueChanged.connect(lambda: self.span_control(1, check_list[1], t_max))
        self.span2.valueChanged.connect(lambda: self.span_control(2, check_list[1]))

        if check_list[2] != IndexConfig.SRD_idx:
            self.listWidget.itemClicked.connect(lambda: self.run_calcul_sta(g_or_r='g', group_idx=check_list[2]))

        if check_list[2] == IndexConfig.ESC_SD_idx:
            self.start_btn.clicked.connect(lambda: self.run_esc_sd(self.esc_sd_mod))
        if check_list[2] == IndexConfig.AD_idx:
            self.start_btn.clicked.connect(lambda: self.run_ad(self.parse))
        if check_list[2] == IndexConfig.SpiD_idx:
            self.span1.valueChanged.disconnect()
            self.span2.valueChanged.disconnect()
            if check_list[1] == 0.3:
                self.to_dur_lbl.setText('to')
                self.span2.setDecimals(2)
                self.span2.setValue(check_list[1])
                self.span1.valueChanged.connect(lambda: self.min_span_control(0, check_list[1], t_max))
                self.span2.valueChanged.connect(lambda: self.min_span_control(1, check_list[1], t_max))
                self.start_btn.clicked.connect(self.run_spid_template)
            elif check_list[1] == 30:
                self.to_dur_lbl.setText('Dur(s): 30s *')
                self.span2.setDecimals(0)
                self.span2.setMinimum(1)
                self.span2.setValue(1)
                self.span1.valueChanged.connect(lambda: self.multi_span_control(0, check_list[1], t_max))
                self.span2.valueChanged.connect(lambda: self.multi_span_control(1, check_list[1], t_max))
                self.start_btn.clicked.connect(self.run_spid_semantics)
        if check_list[2] == IndexConfig.SRD_idx:
            self.span1.valueChanged.disconnect()
            self.span2.valueChanged.disconnect()
            self.span2.setDecimals(2)
            self.span2.setValue(check_list[1])
            self.span1.valueChanged.connect(lambda: self.min_span_control(0, check_list[1], t_max))
            self.span2.valueChanged.connect(lambda: self.min_span_control(1, check_list[1], t_max))
            self.start_btn.clicked.connect(self.run_srd)
        self.check = True
        self.parent.diary.info('Pass Checking')

    def span_control(self, index, span, t_max=None):
        if self.is_updating:
            return
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
        self.is_updating = False

    def min_span_control(self, index, min_span, t_max):
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
        if index == 0:
            self.span1.setValue(min(self.span1.value(), t_max - span))
        if self.span1.value() + self.span2.value() * span > t_max:
            self.span2.setValue(floor((t_max - self.span1.value()) / span))

    def run_calcul_sta(self, g_or_r, group_idx, freq=None, start=None, stop=None):
        from utils.threads import StatisticsThread
        selected_indexed = self.listWidget.selectedIndexes()
        if not selected_indexed:
            return
        ch_idx = selected_indexed[0].row()
        self.statistic_thread = StatisticsThread(g_or_r, group_idx, ch_idx, self.raw.copy(), freq, start, stop)
        self.statistic_thread.res_signal.connect(self.toggle_lbl_to_show)
        self.statistic_thread.start()
        self.parent.diary.info(f'calculate_statistics {g_or_r} + {group_idx}')

    def toggle_lbl_to_show(self, group, g_r, data_list):
        lbl_dict = {
            (IndexConfig.ESC_SD_idx, 'r'): [[self.r_mean, self.r_std, self.r_var],
                                            [self.r_d, self.r_t, self.r_a, self.r_b, self.r_g]],
            (IndexConfig.ESC_SD_idx, 'g'): [[self.g_mean, self.g_std, self.g_var],
                                            [self.g_d, self.g_t, self.g_a, self.g_b, self.g_g]],
            (IndexConfig.AD_idx, 'r'): [[self.r_mean_1, self.r_std_1, self.r_var_1],
                                        [self.r_d_1, self.r_t_1, self.r_a_1, self.r_b_1, self.r_g_1]],
            (IndexConfig.AD_idx, 'g'): [[self.g_mean_1, self.g_std_1, self.g_var_1],
                                        [self.g_d_1, self.g_t_1, self.g_a_1, self.g_b_1, self.g_g_1]],
            (IndexConfig.SpiD_idx, 'r'): [[self.r_mean_2, self.r_std_2, self.r_var_2],
                                          [self.r_d_2, self.r_t_2, self.r_a_2, self.r_b_2, self.r_g_2]],
            (IndexConfig.SpiD_idx, 'g'): [[self.g_mean_2, self.g_std_2, self.g_var_2],
                                          [self.g_d_2, self.g_t_2, self.g_a_2, self.g_b_2, self.g_g_2]],
        }
        label_list = lbl_dict.get((group, g_r), [])
        if label_list is not []:
            self.show_statistics(label_list, data_list)

    @staticmethod
    def show_statistics(label_list, data_list):
        for labels, datum in zip(label_list, data_list):
            for label, data in zip(labels, datum):
                label.setText(f'{data}')

    def run_esc_sd(self, model):
        """ESC_SD 引擎线程启动"""
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.ESC_SD_idx, freq=1000, start=start_time, stop=stop_time)

        self.parent.diary.info('使用ESC+SD引擎线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'model:{model}')

        # 唯一改动：ESCSDThread → ESCSDEngineThread
        self.esc_sd_thread = ESCSDEngineThread(self.raw.copy(), start_time, model)
        self.esc_sd_thread.fm_signal.connect(self.esc_sd_fm.update_img)
        self.esc_sd_thread.feature_signal.connect(self.esc_sd_feature.update_img)
        self.esc_sd_thread.res_signal.connect(self.esc_sd_res.update_img)
        self.esc_sd_thread.error_signal.connect(
            lambda msg: (
                self.process_text_lbl.setText(f'ESC+SD engine error: {msg}'),
                self.parent.diary.warning(msg),
            )
        )
        self.esc_sd_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name=model))
        self.esc_sd_thread.start()

    def ad_unique_widgets(self, show=False):
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
        self.ann_lbl.setVisible(show)
        self.ann_cbx.setVisible(show)

    def run_ad(self, model):
        """AD 引擎线程启动"""
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.AD_idx, freq=1000, start=start_time, stop=stop_time)

        arti_list = self.get_arti_list()
        fb_idx = self.tpm_cbx.currentIndex()
        auto = self.ann_cbx.isChecked()
        self.parent.diary.info('使用AD引擎线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'model1: {model[0]}\n'
                               f'model2: {model[1]}\n'
                               f'artifact list: {arti_list}\n'
                               f'tpm:{fb_idx}\n'
                               f'annotate: {auto}')

        # 唯一改动：ADThread → ADEngineThread
        self.ad_thread = ADEngineThread(
            self.raw.copy(), int(start_time), int(stop_time),
            method=f"{model[0]}_{model[1]}",
            mod1=model[0], mod2=model[1],
            arti_list=arti_list, fb_idx=fb_idx, auto=auto
        )
        self.ad_thread.topo_signal.connect(self.ad_topo.update_img)
        self.ad_thread.res_signal.connect(lambda x: self.ad_res.update_img(img_data=x))
        self.ad_thread.ann_signal.connect(self.annotate_result_overlap)
        self.ad_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name=f'{model[0]}+{model[1]}'))
        self.ad_thread.finish_signal.connect(
            lambda: self.tpm_btn.setEnabled(True) if not self.tpm_btn.isEnabled() else None)
        self.ad_thread.start()

    def get_arti_list(self):
        arti = []
        if self.n_btn.isChecked(): arti.append(0)
        if self.eb_btn.isChecked(): arti.append(1)
        if self.fe_btn.isChecked(): arti.append(2)
        if self.ce_btn.isChecked(): arti.append(3)
        if self.te_btn.isChecked(): arti.append(4)
        if self.u_btn.isChecked(): arti.append(5)
        return arti

    def repaint_tpm(self):
        if self.raw is None:
            return
        raw1 = self.raw.copy().notch_filter(freqs=50)
        raw_filtered = raw1.filter(l_freq=1, h_freq=70)
        raw_data = raw_filtered.get_data()
        n_data = from_numpy(raw_data)
        fb_idx = self.tpm_cbx.currentIndex()
        self.ad_topo.update_img(
            APSD(n_data, self.span1.value(), self.span2.value(), fb_idx)
        )
        self.parent.diary.info(f'tpm:{fb_idx}')

    def run_spid_template(self):
        """SpiD 模板匹配引擎线程启动"""
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.SpiD_idx, freq=500, start=start_time, stop=stop_time)

        auto = self.ann_cbx.isChecked()
        self.parent.diary.info('使用模板匹配引擎线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'stop_time: {stop_time}\n'
                               f'annotate: {auto}')

        self.sd_tem_thread = SpiDEngineThread(
            self.raw.copy(), start_time, stop_time,
            method="Template Matching", auto=auto
        )
        self.sd_tem_thread.res_signal.connect(lambda x, y: self.sd_res.update_img(img_data=x, raw=y))
        self.sd_tem_thread.ann_signal.connect(self.annotate_result_overlap)
        self.sd_tem_thread.swi_signal.connect(lambda x: self.sd_swi.setText('SWI(Spike wave index):' + x))
        self.sd_tem_thread.error_signal.connect(
            lambda msg: (
                self.process_text_lbl.setText(f'SpiD Template Matching error: {msg}'),
                self.parent.diary.warning(f'[Template Matching] {msg}'),
            )
        )
        self.sd_tem_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name='Template Matching'))
        self.sd_tem_thread.start()

    def run_spid_semantics(self):
        """SpiD 语义分割引擎线程启动"""
        if self.model_cbx.currentText() == '':
            return
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = start_time + self.span2.value() * 30
        self.run_calcul_sta(g_or_r='r', group_idx=IndexConfig.SpiD_idx, freq=500, start=start_time, stop=stop_time)

        auto = self.ann_cbx.isChecked()
        self.parent.diary.info('使用语义分割引擎线程进行检测，接口传入参数：\n'
                               f'start time: {start_time}\n'
                               f'stop_time: {stop_time}\n'
                               f'annotate: {auto}')

        self.sd_sem_thread = SpiDEngineThread(
            self.raw.copy(), start_time, stop_time,
            method="Unet+ResNet34", auto=auto
        )
        self.sd_sem_thread.res_signal.connect(lambda x, y: self.sd_res.update_img(img_data=x, raw=y))
        self.sd_sem_thread.ann_signal.connect(self.annotate_result_overlap)
        self.sd_sem_thread.swi_signal.connect(lambda x: self.sd_swi.setText('SWI(Spike wave index):' + x))
        self.sd_sem_thread.error_signal.connect(
            lambda msg: (
                self.process_text_lbl.setText(f'SpiD Semantic Segmentation error: {msg}'),
                self.parent.diary.warning(f'[Semantic Segmentation] {msg}'),
            )
        )
        self.sd_sem_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name='Semantic Segmentation'))
        self.sd_sem_thread.start()

    def annotate_result_overlap(self, ann_list, des_list, start_time, end_time):
        if len(ann_list) > 0:
            ann = self.parent.raw.annotations.copy()
            before_crop = sum(1 for onset, dur in zip(ann.onset, ann.duration) if onset + dur < start_time)
            end_time = end_time - 1 / self.parent.raw.info['sfreq']
            ann.crop(tmin=start_time, tmax=end_time, use_orig_time=False)
            des_in_ann = set(des for des in des_list if des in ann.description)
            if len(des_in_ann) > 0:
                ann_exist = [[onset, dur, des] for onset, dur, des in zip(ann.onset, ann.duration, ann.description)]
                ann_exist_des = [a for a in ann_exist if a[2] in des_in_ann and a[1] != 0]
                if len(ann_exist_des) > 0:
                    temp_ann_list = ann_list.copy()
                    delete_list = []
                    self.parent.diary.info('ann_exist_des:\n'
                                           f'{ann_exist_des}\n'
                                           'temp_ann_list:\n'
                                           f'{temp_ann_list}')
                    for b in temp_ann_list:
                        cur_b = b

                        def overlap(new_ann, new_b):
                            for i, a in enumerate(new_ann):
                                if new_b[2] != a[2]:
                                    continue
                                if new_b[0] + new_b[1] < a[0]:
                                    ann_list.append(new_b)
                                    ann_list.remove(cur_b)
                                    return True
                                if new_b[0] == a[0] > start_time and new_b[0] + new_b[1] == a[0] + a[1] < end_time:
                                    ann_list.remove(cur_b)
                                    return True
                                if (new_b[0] + new_b[1] >= a[0] and new_b[0] <= a[0] + a[1]) \
                                        or new_b[0] == a[0] == start_time \
                                        or new_b[0] + new_b[1] == a[0] + a[1] == end_time:
                                    delete_idx = ann_exist.index(a) + before_crop
                                    delete_list.append(delete_idx)
                                    new_ann.remove(a)
                                    ann_exist_des.remove(a)
                                    a_start = self.parent.raw.annotations.onset[delete_idx]
                                    a_end = a_start + self.parent.raw.annotations.duration[delete_idx]
                                    onset = min(new_b[0], a_start)
                                    new_b = [onset, max(new_b[0] + new_b[1], a_end) - onset, new_b[2]]
                                    if len(new_ann) > 0:
                                        if overlap(new_ann, new_b):
                                            return
                                    else:
                                        ann_list.append(new_b)
                                        ann_list.remove(cur_b)
                                        return True
                                if i == len(new_ann) - 1:
                                    ann_list.append(new_b)
                                    ann_list.remove(cur_b)
                                    return True

                        overlap(ann_exist_des.copy(), cur_b)
                    if len(delete_list) > 0:
                        delete_list.sort(reverse=True)
                        for idx in delete_list:
                            self.parent.raw.annotations.delete(idx)

        if len(ann_list) > 0:
            for a in ann_list:
                self.parent.raw.annotations.append(a[0], a[1], a[2])
            self.parent.plot_eeg(self.parent.raw)
            self.parent.get_ann()
            self.raw = self.parent.raw.copy()
        self.set_process_txt(10)

    def run_srd(self):
        """SRD 引擎线程启动"""
        if self.model_cbx.currentText() == '':
            return
        selected_indexed = self.listWidget.selectedIndexes()
        if not selected_indexed:
            self.set_process_txt(14)
            return
        ch_idx = selected_indexed[0].row()
        self.set_process_txt(7)
        start_time = self.span1.value()
        stop_time = self.span2.value()

        self.parent.diary.info('使用SRD引擎线程进行检测，接口传入参数：\n'
                               f'ch_idx: {ch_idx}\n'
                               f'start time: {start_time}\n'
                               f'stop_time: {stop_time}\n')

        self.hfo_thread = SRDEngineThread(self.raw.copy(), ch_idx, start_time, stop_time)
        self.hfo_thread.res_signal.connect(lambda x: self.plot_hfo(x, start_time))
        self.hfo_thread.finish_signal.connect(lambda: self.set_process_txt(8, model_name='SRD'))
        self.hfo_thread.error_signal.connect(
            lambda msg: (
                self.process_text_lbl.setText(f'SRD error: {msg}'),
                self.parent.diary.warning(f'[SRD] {msg}'),
            )
        )
        self.hfo_thread.start()

    def plot_hfo(self, merged_raw, start_time):
        try:
            while self.hfo_layout.count():
                item = self.hfo_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            self.hfo_layout.addWidget(show_plot(merged_raw, start_time))
            self.parent.diary.info(f'SRD绘图完成，HFO事件数: {len(merged_raw.annotations)}')
        except Exception as e:
            self.parent.diary.warning(f'SRD绘图失败: {e}')
            self.process_text_lbl.setText(f'SRD绘图失败: {e}')

    def vd_ini(self):
        """VD 初始化（保持原样，因为 VD 模型加载逻辑不同）"""
        self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('start')))
        self.vd_stop_btn.setIcon(QIcon(AddressConfig.get_icon_adr('stop')))
        self.list_btn.setIcon(QIcon(AddressConfig.get_icon_adr('list')))

        # VD 引擎线程（内部已含模型注入逻辑）
        self.vd_thread = VDEngineThread([], "", None, None)
        self.vd_thread.img_signal.connect(lambda x: self.vd_show_video_img(x, self.output_lbl))
        self.vd_thread.res_signal.connect(self.vd_res.addItem)
        self.vd_thread.percent_signal.connect(lambda x: self.progressBar.setValue(x))
        self.vd_thread.normal_finish_signal.connect(self.vd_stop)
        self.vd_thread.normal_finish_signal.connect(lambda: self.set_process_txt(8, model_name='VD'))
        self.vd_thread.abnormal_finish_signal.connect(lambda: self.set_process_txt(11))

        # VD 模型加载线程（保持原 VDModelThread，注入到 VDEngineThread）
        self.vd_mod_thread = VDModelThread(self.vd_thread.cfg)
        self.vd_mod_thread.mod_signal.connect(self.vd_thread.update_mod)
        self.vd_mod_thread.finish_signal.connect(lambda: self.set_process_txt(12))
        self.vd_mod_thread.finish_signal.connect(self.vd_activate)
        self.vd_mod_thread.start()

    def vd_hide_widgets(self, show=True):
        self.gb_1.setVisible(show)
        self.gb_2.setVisible(show)
        self.gb_3.setVisible(show)

    def vd_setting_box(self):
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
        self.vd_export_btn.clicked.connect(self.vd_export_res)
        self.input_adr_le.setEnabled(True)
        self.output_adr_le.setEnabled(True)
        self.input_adr_le.textChanged.connect(self.vd_path_setting)
        self.output_adr_le.textChanged.connect(self.vd_path_setting)
        self.vd_save_cbx.setEnabled(True)
        self.iou_sb.setEnabled(True)
        self.iou_sd.setEnabled(True)
        self.conf_sb.setEnabled(True)
        self.conf_sd.setEnabled(True)
        self.vd_save_cbx.clicked.connect(self.vd_save_setting)
        self.iou_sb.valueChanged.connect(lambda x: self.vd_change_value(x, 'iou_sb'))
        self.iou_sd.valueChanged.connect(lambda x: self.vd_change_value(x, 'iou_sd'))
        self.conf_sb.valueChanged.connect(lambda x: self.vd_change_value(x, 'conf_sb'))
        self.conf_sd.valueChanged.connect(lambda x: self.vd_change_value(x, 'conf_sd'))
        self.vd_start_btn.clicked.connect(self.vd_run_or_continue)
        self.vd_stop_btn.clicked.connect(self.vd_stop)

    def vd_path_setting(self):
        self.vd_stop()
        input_list = self.input_adr_le.input_list
        if len(input_list) > 0:
            self.vd_thread.update_adr(input_list, self.output_adr_le.text())

    def vd_save_setting(self):
        if self.vd_save_cbx.isChecked():
            self.vd_thread.save = True
        else:
            self.vd_thread.save = False

    def vd_change_value(self, x, flag):
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
        if self.input_adr_le.text() == '':
            QMessageBox.warning(self, 'Warning', 'Please select the input file(s)!')
            return
        self.vd_thread.jump_out = False
        if self.vd_start_btn.isChecked():
            self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('pause')))
            self.vd_thread.is_continue = True
            self.set_process_txt(7)
            if not self.vd_thread.isRunning():
                self.vd_res.clear()
                self.parent.diary.info('使用VD引擎线程进行检测，接口传入参数：\n'
                                       f'input list: {self.vd_thread.input_list}\n'
                                       f'output address: {self.vd_thread.output_adr}\n'
                                       f'save: {self.vd_thread.save}')
                self.vd_thread.start()
        else:
            self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('start')))
            self.set_process_txt(13)
            self.vd_thread.is_continue = False

    def vd_stop(self):
        self.vd_start_btn.setChecked(False)
        self.vd_start_btn.setIcon(QIcon(AddressConfig.get_icon_adr('start')))
        self.vd_thread.jump_out = True

    @staticmethod
    def vd_show_video_img(img, label):
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
        num_receivers1 = self.start_btn.receivers(self.start_btn.clicked)
        if num_receivers1 > 0:
            self.start_btn.clicked.disconnect()
        num_receivers2 = self.span1.receivers(self.span1.valueChanged)
        if num_receivers2 > 0:
            self.span1.valueChanged.disconnect()
        num_receivers3 = self.span2.receivers(self.span2.valueChanged)
        if num_receivers3 > 0:
            self.span2.valueChanged.disconnect()
        num_receivers4 = self.listWidget.receivers(self.listWidget.itemClicked)
        if num_receivers4 > 0:
            self.listWidget.itemClicked.disconnect()

    def show_local_img(self, label, img_path):
        img = QImage()
        if img.load(img_path):
            pixmap = QPixmap.fromImage(img)
            label.setPixmap(pixmap)
        else:
            self.parent.diary.warning('Unable to load image.')

    def set_process_txt(self, status_code, service_name=None, sfreq=None, min_time=None, model_name=None):
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
        self.parent.extend_services_form = None
        event.accept()
        self.parent.diary.debug('ExtendServicesForm —> MainForm')
