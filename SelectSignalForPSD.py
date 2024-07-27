from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox

ui_select_signal_for_psd = uic.loadUiType('ui/select_signal_for_psd.ui')[0]


class SelectSignalForPSD(QWidget, ui_select_signal_for_psd):
    def __init__(self, parent=None):
        super(SelectSignalForPSD, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()

    def init_ui(self):
        self.ok_btn.clicked.connect(self.select_ch)
        self.select_ch_cbx.addItems(self.parent.raw.info['ch_names'])

    def select_ch(self):
        """
        list: [ch_name, fmin, fmax]
        """
        if self.select_ch_cbx.currentText() == '<select channel>':
            return
        try:
            self.parent.diary.info(f'PSD channel: {self.select_ch_cbx.currentText()}\n'
                                   f'Frequency: {self.x1.value()} to {self.x2.value()}Hz')
            self.parent.plot_psd([self.select_ch_cbx.currentText(), self.x1.value(), self.x2.value()])
            self.close()
            self.parent.diary.debug('SelectSignalForPSD —> MainForm')
        except Exception as e:
            self.parent.diary.error('<FAILED>' + str(e))
            QMessageBox.warning(self, 'Warning', str(e))
