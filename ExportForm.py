from PyQt5 import uic
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog
from mne import export, Annotations
from numpy import arange
from os.path import join

ui_export_form = uic.loadUiType('ui/export_edf_form.ui')[0]


class ExportForm(QWidget, ui_export_form):
    def __init__(self, parent=None):
        super(ExportForm, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.init_ui()
        self.t_max = self.parent.raw.times[-1]
        self.x1.setMaximum(self.t_max)
        self.raw = self.parent.raw.copy()
        self.min_time = 1 / self.parent.raw.info['sfreq']
        self.all_lbl.setText(f'All ({self.t_max+self.min_time}s)')

    def init_ui(self):
        self.ok_btn.clicked.connect(self.export_control)
        self.x2.valueChanged.connect(lambda: self.autoset_span(self.x2))
        self.time_span.valueChanged.connect(lambda: self.autoset_span(self.time_span))

    def autoset_span(self, qdsb):
        if qdsb.value() > self.t_max:
            qdsb.setValue(self.t_max)

    def export_all_edf(self, adr):
        try:
            self.parent.diary.info('Export EDF: All')
            export.export_raw(adr, self.raw, overwrite=True)
        except Exception as e:
            self.parent.diary.error('<FAILED> ' + str(e))
            QMessageBox.warning(self, 'Warning: ', str(e))

    def export_part_edf(self, adr):
        """
        保存选中时间段内的事件
        """
        try:
            ann = self.raw.annotations
            ann.crop(tmin=self.x1.value(), tmax=self.x2.value() - self.min_time,
                     use_orig_time=False)
            if len(ann.onset) != 0:
                ann.onset = [onset - self.x1.value() for onset in ann.onset]
            raw = self.raw.crop(self.x1.value(), self.x2.value() - self.min_time)
            raw.set_annotations(Annotations([], [], []))  # 先清除，后追加
            raw.annotations.append(ann.onset, ann.duration, ann.description)
            self.parent.diary.info(f'Export EDF: {self.x1.value()} to {self.x2.value()}s\n'
                                   f'Adr: {adr}\n'
                                   f'Annotation Onset: {ann.onset}\n'
                                   f'Annotation Duration: {ann.duration}\n'
                                   f'Annotation Description: {ann.description}')
            export.export_raw(adr, raw, overwrite=True)
        except Exception as e:
            self.parent.diary.error('<FAILED> ' + str(e))
            QMessageBox.warning(self, 'Warning: ', str(e))

    def clip_edf(self, span, folder_adr):
        """
        不保存选中时间段内的事件
        """
        try:
            start_times = arange(0, self.t_max, span)
            end_times = start_times + span - self.min_time
            end_times[-1] = self.t_max

            for i, (start, end) in enumerate(zip(start_times, end_times)):
                clip = self.raw.copy().crop(start, end)
                # clip.set_annotations(Annotations([], [], []))
                output_path = join(folder_adr, f'clip_{i + 1}.edf')
                export.export_raw(output_path, clip, overwrite=True)
            self.parent.diary.info(f'Export EDF: per {span} s\n'
                                   f'Adr: {folder_adr}')
        except Exception as e:
            self.parent.diary.error('<FAILED> ' + str(e))
            QMessageBox.warning(self, 'Warning: ', str(e))

    def get_adr(self, folder=False) -> str:
        if not folder:
            adr, _ = QFileDialog.getSaveFileName(self, 'Export .edf', '.', 'EDF files (*.edf)')
        else:
            adr = QFileDialog.getExistingDirectory(self, "Select Save Folder")  # 单个文件夹
        return adr if adr else None

    def export_control(self):
        if self.all_btn.isChecked():
            adr = self.get_adr()
            if adr is None:
                return
            self.export_all_edf(adr)
        elif self.part_btn.isChecked():
            if self.x2.value() <= self.x1.value():
                QMessageBox.warning(self, 'Warning', 'The current time span is invalid!')
                return
            adr = self.get_adr()
            if adr is None:
                return
            self.export_part_edf(adr)
        else:
            adr = self.get_adr(True)
            if adr is None:
                return
            self.clip_edf(self.time_span.value(), adr)

        self.close()
        self.parent.diary.debug('ExportForm —> MainForm')
