from PyQt5 import uic
from PyQt5.QtWidgets import QWidget

ui_filter_options_form = uic.loadUiType('ui/filter_options_form.ui')[0]


class FilterOptionsForm(QWidget, ui_filter_options_form):
    def __init__(self, parent=None):
        super(FilterOptionsForm, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()

    def init_ui(self):
        self.ok_btn.clicked.connect(self.check)
        self.lp_cbx.clicked.connect(self.reset_bp)
        self.hp_cbx.clicked.connect(self.reset_bp)
        self.bp_cbx.clicked.connect(self.reset_lp_hp)

    def reset_bp(self):
        self.bp_cbx.setChecked(False)

    def reset_lp_hp(self):
        self.lp_cbx.setChecked(False)
        self.hp_cbx.setChecked(False)

    def check(self):
        if self.lp_cbx.isChecked():
            self.parent.fi.do_lp = 1
            self.parent.fi.do_bp = 0
            self.parent.fi.hf = self.lp_box.value()
        else:
            self.parent.fi.do_lp = 0
        if self.hp_cbx.isChecked():
            self.parent.fi.do_hp = 1
            self.parent.fi.do_bp = 0
            self.parent.fi.lf = self.hp_box.value()
        else:
            self.parent.fi.do_hp = 0
        if self.n_cbx.isChecked():
            self.parent.fi.do_notch = 1
            self.parent.fi.notch = self.n_box.value()
        else:
            self.parent.fi.do_notch = 0
        if self.bp_cbx.isChecked():
            self.parent.fi.do_bp = 1
            self.parent.fi.do_hp = self.parent.fi.do_lp = 0
            self.parent.fi.bp_hf = self.bp_box2.value()
            self.parent.fi.bp_lf = self.bp_box1.value()
        else:
            self.parent.fi.do_bp = 0
        if self.parent.filter_cbx.isChecked():
            self.parent.filter_cbx.setChecked(False)
            self.parent.filter_cbx.setText('on')
        self.close()
