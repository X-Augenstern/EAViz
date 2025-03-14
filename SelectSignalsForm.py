from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox
# from utils.config import AddressConfig
# from subprocess import Popen
from mne import io
from utils.config import ChannelEnum
from utils.edf import EdfUtil

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
        self.listWidget.addItems(ChannelEnum.CH21.value)
        # self.yaml_btn.clicked.connect(self.open_yaml)

    def select_chns(self, service):
        chns = None
        self.listWidget.clearSelection()
        if service == 'SeiD/ESA':
            chns = ChannelEnum.CH21.value
        elif service == 'AD/SD':
            chns = ChannelEnum.CH19.value
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
            raw = EdfUtil.normalize_edf(self.adr, selected_list, self.parent.diary)
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
