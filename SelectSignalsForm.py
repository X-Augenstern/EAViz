from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox
# from utils.config import AddressConfig
# from subprocess import Popen
from mne import io

ui_select_signals_form = uic.loadUiType('ui/select_signals_form.ui')[0]


class SelectSignalsForm(QWidget, ui_select_signals_form):
    def __init__(self, parent=None, adr=None, preload=False):
        super(SelectSignalsForm, self).__init__()
        self.parent = parent
        self.adr = adr
        self.setupUi(self)
        self.init_ui()
        self.preload = preload

    def init_ui(self):
        self.ok_btn.clicked.connect(self.load_raw)
        self.esa_btn.clicked.connect(lambda: self.select_chns('SeiD/ESA'))
        self.sd_btn.clicked.connect(lambda: self.select_chns('AD/SD'))
        self.listWidget.addItems(self.parent.channel_config.SeiD_ESA_channels)
        # self.yaml_btn.clicked.connect(self.open_yaml)

    def select_chns(self, service):
        chns = None
        self.listWidget.clearSelection()
        if service == 'SeiD/ESA':
            chns = self.parent.channel_config.SeiD_ESA_channels
        elif service == 'AD/SD':
            chns = self.parent.channel_config.AD_SD_channels
        for ch in chns:
            for i in range(self.listWidget.count()):
                if ch == self.listWidget.item(i).text():
                    self.listWidget.item(i).setSelected(True)

    def load_raw(self):
        selected_items = self.listWidget.selectedItems()
        if not selected_items:
            return
        # selected_list = [item.text() for item in selected_items]  # selected_list中元素会按照选择顺序排序
        selected_list = [self.listWidget.item(i).text() for i in range(self.listWidget.count())
                         if self.listWidget.item(i).isSelected()]  # 从上到下遍历确保顺序不会出错
        self.parent.diary.info(f'selected_list(before): {selected_list}')
        try:
            raw = io.read_raw_edf(self.adr, preload=True)
            # raw = io.read_raw_eeglab(self.adr)
            raw_chns = raw.info['ch_names']
            self.parent.diary.info(f'EDF_adr: {self.adr}\n'
                                   f'EDF_chns: {raw_chns}')
            # 获得统一通道名：selected_list与其在edf中的映射：mapping_list（目标：数量相等，名字对应）
            tmp_list = selected_list.copy()
            mapping_list = []
            for key in tmp_list:
                tmp = []
                for ch_name in raw_chns:
                    if key in ch_name:
                        tmp.append(ch_name)
                if len(tmp) == 0:  # 选择 > 存在
                    selected_list.remove(key)
                elif len(tmp) == 1:  # 选择 = 存在
                    mapping_list.extend(tmp)
                else:  # 选择 < 存在：存在相近的通道名，则只保留最短的
                    tmp = [ch_name for ch_name in tmp if len(ch_name) == min(len(ch) for ch in tmp)]
                    if len(tmp) == 1:
                        mapping_list.extend(tmp)
                    else:  # 存在两条名字相同的通道：直接报异常
                        mapping_list = []
                        break
            self.parent.diary.info(f'selected_list(after) = {selected_list}')
            self.parent.diary.info(f'mapping_list = {mapping_list}')
            self.parent.diary.info(f'len == ? {len(mapping_list) == len(selected_list)}')
            assert len(mapping_list) == len(selected_list)
            # 按映射表调整通道顺序，去除多余的通道
            raw.reorder_channels(mapping_list)
            # 修改通道名
            dic = {}
            for i in range(len(mapping_list)):
                dic[mapping_list[i]] = selected_list[i]
            raw.rename_channels(dic)
            # raw.pick(picks=selected_list)  # 会改变raw的通道，且通道名称会按list的指定顺序排列
            if not self.preload:
                self.parent.get_raw(raw=raw, eeg_list=selected_list, adr=self.adr)
            else:
                self.parent.get_raw(raw=raw, eeg_list=selected_list, preload=self.preload)
            self.close()
            self.parent.diary.debug('SelectSignalsForm —> MainForm')
        except Exception as e:
            self.parent.diary.error('<FAILED>' + str(e))
            QMessageBox.warning(self, 'Warning', 'Invalid .edf!\n' + str(e))
        # self.parent.topo_ch_cbx.addItems(list(set(self.chs) & set(self.parent.topomap_chs_list)))

    # @staticmethod
    # def open_yaml():
    #     try:
    #         Popen(['start', AddressConfig.yaml_adr], shell=True)
    #     except Exception as e:
    #         print(f"无法打开文件: {e}")
