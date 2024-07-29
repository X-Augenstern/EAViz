from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox

ui_select_signals_for_psd = uic.loadUiType('ui/select_signals_for_psd.ui')[0]


class SelectSignalsForPSD(QWidget, ui_select_signals_for_psd):
    def __init__(self, parent=None):
        super(SelectSignalsForPSD, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()

    def init_ui(self):
        self.ok_btn.clicked.connect(self.select_chs)
        self.listWidget.addItems(self.parent.raw.info['ch_names'])

    def select_chs(self):
        """
        list: [[ch_name1, ch_name2, ...], fmin, fmax]
        """
        selected_items = self.listWidget.selectedItems()
        if not selected_items:
            return
        selected_list = [self.listWidget.item(i).text() for i in range(self.listWidget.count())
                         if self.listWidget.item(i).isSelected()]  # 从上到下遍历确保顺序不会出错
        try:
            self.parent.diary.info(f'PSD channels: {selected_list}\n'
                                   f'Frequency: {self.x1.value()} to {self.x2.value()}Hz')
            self.parent.plot_psd([selected_list, self.x1.value(), self.x2.value()])
            self.close()
            self.parent.diary.debug('SelectSignalsForPSD —> MainForm')
        except Exception as e:
            self.parent.diary.error('<FAILED>' + str(e))
            QMessageBox.warning(self, 'Warning', str(e))
