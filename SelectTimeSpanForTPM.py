from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox

ui_select_time_span_for_tpm = uic.loadUiType('ui/select_time_span_for_tpm.ui')[0]


class SelectTimeSpanForTPM(QWidget, ui_select_time_span_for_tpm):
    def __init__(self, parent=None):
        super(SelectTimeSpanForTPM, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()
        self.t_max = self.parent.raw.times[-1]
        self.x1.setMaximum(self.t_max)

    def init_ui(self):
        self.ok_btn.clicked.connect(self.select_span)
        self.x2.valueChanged.connect(lambda: self.autoset_span(self.t_max))

    def autoset_span(self, t_max):
        if self.x2.value() >= t_max:
            self.x2.setValue(t_max)

    def select_span(self):
        """
        list: [tmin, tmax]
        """
        if self.span_btn.isChecked():
            if self.x2.value() <= self.x1.value():
                QMessageBox.warning(self, 'Warning', 'The current span is invalid!')
                return
            self.parent.diary.info(f'TPM time: {self.x1.value()} to {self.x2.value()}s')
            self.parent.plot_tpm([self.x1.value(), self.x2.value()])
        else:
            self.parent.diary.info('TPM time: All')
            self.parent.plot_tpm()
        self.close()
        self.parent.diary.debug('SelectTimeSpanForTPM —> MainForm')
