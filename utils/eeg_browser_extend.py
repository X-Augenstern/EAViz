from PyQt5.QtCore import QObject, QEvent, Qt
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from mne.viz import use_browser_backend
from mne.viz.utils import _merge_annotations
from mne.annotations import _sync_onset
from mne_qt_browser._pg_figure import AnnotRegion, _dark_dict, _rgb_to_lab, _lab_to_rgb
from utils.config import ThemeColorConfig
from numpy import diff, array
from copy import copy


class RawViewBoxEscapeKeyFilter(QObject):
    """
    To filter the escape behaviour of the innermost widget(RawViewBox).

    Trace:
        1335、3221
    """

    def __init__(self, main_form):
        super().__init__()
        self.main_form = main_form  # hold MainForm's ref

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            if self.main_form.isFullScreen():
                self.exit_full_screen()
                # print("Exited full-screen mode from RawViewBox")
                return True  # 阻止事件继续传递
            else:
                return True
        return super().eventFilter(obj, event)

    def full_screen(self):
        for gb in self.main_form.groupbox_list:
            gb.setVisible(False)
        self.main_form.showFullScreen()

    def exit_full_screen(self):
        self.main_form.showNormal()
        for gb in self.main_form.groupbox_list:
            gb.setVisible(True)


class EEGBrowserManager:
    window_size = 5  # window内采样时间（duration）
    n_channels = 10  # window内channel数
    color = None
    amplitude = 600  # 600e-6 = 600*10^-6 = 600*10**-6  !=  600*10e-6 = 600*10*10^-6
    show = False
    use_opengl = True

    def __init__(self, main_form):
        self.eeg_browser = None
        self.filter = RawViewBoxEscapeKeyFilter(main_form)
        self.bg_color = ThemeColorConfig.get_eeg_bg()

    def get_eeg_browser(self):
        return self.eeg_browser

    def _get_cur_x(self):
        """
        获取当前EEG视图左、右侧的时间
        """
        return self.eeg_browser.mne.viewbox.viewRange()[0]

    def _get_cur_xmin(self):
        return self._get_cur_x()[0] if self.eeg_browser else 0

    def _get_cur_dur(self):
        if self.eeg_browser:
            x_min, x_max = self._get_cur_x()
            return x_max - x_min
        else:
            return EEGBrowserManager.window_size

    def _get_cur_y(self):
        return self.eeg_browser.mne.viewbox.viewRange()[1]

    def _get_nch(self):
        if self.eeg_browser:
            y_min, y_max = self._get_cur_y()  # start from 0
            return int(y_max - y_min - 1)
        else:
            return EEGBrowserManager.n_channels

    def _get_cur_amp(self):
        if self.eeg_browser:
            cur_amp_txt = tuple(t.toPlainText() for t in self.eeg_browser.mne.scalebar_texts.values())
            assert len(cur_amp_txt) == 1
            return float(cur_amp_txt[0][:-2])
        else:
            return EEGBrowserManager.amplitude

    def create_eeg_browser(self, raw, bg_color):
        """
        mne-qt-browser
        """
        if bg_color and bg_color != self.bg_color:
            self.bg_color = bg_color

        window_size = self._get_cur_dur()
        xmin = self._get_cur_xmin()
        nch = self._get_nch()
        amp = self._get_cur_amp()

        if self.eeg_browser:
            if window_size >= raw.times[-1]:
                window_size = EEGBrowserManager.window_size
            if xmin >= raw.times[-1]:
                xmin = raw.times[-1] - window_size
                if xmin <= 0:
                    xmin = 0
            self.eeg_browser.close()

        # with 语句的作用是管理资源的进入和退出，而不会影响在 with 块中创建的变量的生命周期
        with use_browser_backend('qt'):
            self.eeg_browser = raw.plot(duration=window_size, start=xmin, n_channels=nch, scalings=amp * 10 ** -6 / 2,
                                        show=EEGBrowserManager.show, use_opengl=EEGBrowserManager.use_opengl,
                                        color=EEGBrowserManager.color, bgcolor=self.bg_color,
                                        theme=ThemeColorConfig.theme)
            self.set_global_matplotlib_color()
            self.eeg_browser.mne.viewbox.installEventFilter(self.filter)

    def change_amplitude(self, flag, tar_amp):
        """
        Trace:
            1630 - update_value() \n
            3750 - _add_scalebars() \n
            3799 - scale_all() \n
            4994 - _get_scale_bar_texts() \n
        Args:
            flag: 'p'(plus) | 'm'(minus) | 'c'(custom)
            tar_amp: target amplitude
        """
        if self.eeg_browser is None:
            return

        if flag == 'p':
            self.eeg_browser.scale_all(step=5 / 4)
        elif flag == 'm':
            self.eeg_browser.scale_all(step=4 / 5)
        elif flag == 'c':
            self.eeg_browser.scale_all(step=self._get_cur_amp() / tar_amp)

    def change_duration(self, tar_dur):
        """
        Trace:
            3877 - change_duration()
        Args:
            tar_dur: target duration(window size)
        """
        if self.eeg_browser is None:
            return

        xmin = self._get_cur_xmin()
        min_dur = 3 * diff(self.eeg_browser.mne.inst.times[:2])[0]
        xmax = xmin + tar_dur
        if xmax - xmin < min_dur:
            xmax = xmin + min_dur
        if xmax > self.eeg_browser.mne.xmax:  # true max
            xmax = self.eeg_browser.mne.xmax
            xmin = xmax - tar_dur
        if xmin < 0:
            xmin = 0
        self.eeg_browser.mne.ax_hscroll.update_duration()
        self.eeg_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def change_num_signals(self, flag):
        """
        Trace:
            3910 - change_nchan()
        Args:
            flag: 'p'(plus) | 'm'(minus)
        """
        if self.eeg_browser is None:
            return

        if flag == 'p':
            self.eeg_browser.change_nchan(step=1)
        elif flag == 'm':
            self.eeg_browser.change_nchan(step=-1)

    def jump_to(self, start):
        """
        Trace:
            3877 - change_duration()
        Args:
            start: jump to specific time
        """
        if self.eeg_browser is None:
            return

        cur_dur = self._get_cur_dur()
        min_dur = 3 * diff(self.eeg_browser.mne.inst.times[:2])[0]
        xmax = start + cur_dur
        if cur_dur < min_dur:
            xmax = start + min_dur
        if xmax > self.eeg_browser.mne.xmax:  # true max
            xmax = self.eeg_browser.mne.xmax
        self.eeg_browser.mne.ax_hscroll.update_duration()
        self.eeg_browser.mne.plt.setXRange(start, xmax, padding=0)

    def full_screen(self):
        if self.eeg_browser is None:
            return

        self.filter.full_screen()
        self.eeg_browser.mne.viewbox.setFocus()

    def _register_des(self, des):
        """
        Trace:
            2572 - _add_description_dlg()
        Args:
            des: new description
        """
        if self.eeg_browser is None:
            return

        ann_dock = self.eeg_browser.mne.fig_annotation
        if des not in self.eeg_browser.mne.new_annotation_labels:
            ann_dock._add_description(des)

    def create_ann(self, des, onset, dur):
        """
        Trace:
            1347 - mouseDragEvent()
        Args:
            des: new description
            onset: new onset
            dur: new dur
        """
        if self.eeg_browser is None:
            return

        self._register_des(des)
        viewbox = self.eeg_browser.mne.viewbox
        offset = onset + dur

        # Create region
        new_region = AnnotRegion(  # 创建绘制区域
            viewbox.mne,
            description=des,
            values=(onset, offset),
            weakmain=viewbox.weakmain  # 弱引用主类，避免循环引用
        )

        # Add to annotations
        sync_onset = _sync_onset(viewbox.mne.inst, onset, inverse=True)  # 将界面中的时间点 onset 转换为 MNE 的时间基准
        # 将新的时间段合并到 annotations 数据结构中，处理可能的重叠
        _merge_annotations(sync_onset, sync_onset + dur, des, viewbox.mne.inst.annotations)

        # Add to regions/merge regions
        merge_values = [onset, offset]
        rm_regions = list()
        for region in viewbox.mne.regions:
            if region.description != des:
                continue
            values = region.getRegion()  # (0, 5)
            if any(onset <= val <= offset for val in values):  # 寻找与新区域时间段重叠的区域
                merge_values += values  # [2, 7, 0, 5]
                rm_regions.append(region)  # (0, 5)
        if len(merge_values) > 2:  # 包含多个时间段，表示存在重叠
            new_region.setRegion((min(merge_values), max(merge_values)))  # (0, 5) -> (0, 7)
        for rm_region in rm_regions:
            viewbox.weakmain()._remove_region(rm_region, from_annot=False)  # 移除旧区域
        viewbox.weakmain()._add_region(  # 添加新区域
            onset,
            dur,
            des,
            region=new_region
        )
        new_region.select(True)  # 设为选中状态
        new_region.setZValue(2)  # 显示优先级

        # Update Overview-Bar
        viewbox.mne.overview_bar.update_annotations()  # 更新，确保新的注释信息在全局概览中可见

    def rename_des(self, idx, des):
        """
        Trace:
            2635 - _edit_description_dlg() \n
            2459 - _edit() \n
            2609 - _edit_description_selected() \n
            2314 - update_description()
        Args:
            idx: index of annotation
            des: new description
        """
        if self.eeg_browser is None:
            return

        ann_dock = self.eeg_browser.mne.fig_annotation

        # Update regions & annotations
        old_des = ann_dock.mne.inst.annotations.description[idx]
        ann_dock.mne.inst.annotations.description[idx] = des
        ann_dock.mne.regions[idx].update_description(des)

        # Update containers with annotation-attributes
        if des not in ann_dock.mne.new_annotation_labels:
            ann_dock.mne.new_annotation_labels.append(des)
        ann_dock.mne.visible_annotations[des] = copy(ann_dock.mne.visible_annotations[old_des])
        if old_des not in ann_dock.mne.inst.annotations.description:
            ann_dock.mne.new_annotation_labels.remove(old_des)
            ann_dock.mne.visible_annotations.pop(old_des)
            ann_dock.mne.annotation_segment_colors[des] = ann_dock.mne.annotation_segment_colors.pop(old_des)

        # Update related widgets
        ann_dock.weakmain()._setup_annotation_colors()
        ann_dock._update_regions_colors()
        ann_dock._update_description_cmbx()
        ann_dock.mne.overview_bar.update_annotations()

    def delete_des(self, des):
        """
        Trace:
            2645 - _remove_description()
        Args:
            des: annotation to delete
        """
        if self.eeg_browser is None:
            return

        ann_dock = self.eeg_browser.mne.fig_annotation

        # Remove regions
        exist_des = False
        for rm_region in [r for r in ann_dock.mne.regions if r.description == des]:
            exist_des = True
            rm_region.remove()
        if not exist_des:
            return

        # Remove from descriptions
        ann_dock.mne.new_annotation_labels.remove(des)
        ann_dock._update_description_cmbx()

        # Remove from visible annotations
        ann_dock.mne.visible_annotations.pop(des)

        # Remove from color-mapping
        if des in ann_dock.mne.annotation_segment_colors:
            ann_dock.mne.annotation_segment_colors.pop(des)

        # Set first description in Combo-Box to current description
        if ann_dock.description_cmbx.count() > 0:
            ann_dock.description_cmbx.setCurrentIndex(0)
            ann_dock.mne.current_description = ann_dock.description_cmbx.currentText()

    def clear_ann(self):
        if self.eeg_browser is None:
            return

        des_list = list(set(annot['description'] for annot in self.eeg_browser.mne.inst.annotations))
        for des in des_list:
            self.delete_des(des)

    @staticmethod
    def _to_dark_mode_color(hex_color, invert=False):
        """
        转换输入十六进制颜色为黑暗模式下的对应颜色。

        Trace:
            182 - _get_color_cached()

        Parameters:
            hex_color (str): 输入的十六进制颜色代码（如 `"#FFFFFF"`）。
            invert (bool): 是否对颜色进行亮度反转。

        Returns:
            str: 黑暗模式下的十六进制颜色代码。
        """
        # 将颜色解析为 RGBA 格式，范围 0-1
        rgba = array(to_rgba(hex_color))
        rgb = tuple((rgba[:3] * 255).astype(int))  # 转换为 0-255 整数 RGB

        # 1. 查找预定义映射表
        if rgb in _dark_dict:
            dark_rgb = array(_dark_dict[rgb]) / 255.0  # 映射的 RGB 转换为 0-1
        else:
            # 2. 自动计算反转颜色（基于亮度）
            if invert:
                lab = _rgb_to_lab(rgba[:3])  # 调用已有的 `_rgb_to_lab`
                lab[0] = 100.0 - lab[0]  # 反转亮度
                dark_rgb = _lab_to_rgb(lab)  # 调用已有的 `_lab_to_rgb`
            else:
                dark_rgb = rgba[:3]  # 保留原始颜色

        # 转换为十六进制格式
        dark_rgb_255 = (dark_rgb * 255).astype(int)
        return "#{:02X}{:02X}{:02X}".format(*dark_rgb_255)

    def set_global_matplotlib_color(self):
        invert = True if ThemeColorConfig.theme == "dark" else False
        bg_color = self._to_dark_mode_color(self.bg_color, invert)
        rcParams['figure.facecolor'] = bg_color
        rcParams['axes.facecolor'] = bg_color
