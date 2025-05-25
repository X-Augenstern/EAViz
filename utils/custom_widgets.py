from os import path, startfile, environ
from sys import platform
from platform import system
from subprocess import Popen, run
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QSizePolicy, QLineEdit, QFileDialog, QPushButton, \
    QVBoxLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QCursor, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer, QSize
from pyqtgraph.dockarea import Dock
from utils.config import AddressConfig, ThemeColorConfig
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.pyplot import close
from matplotlib import rcParams


class SaveButton(QPushButton):
    """
    Overwrite QPushButton to cater to the save figure requirement.
    """

    def __init__(self, parent=None, use_custom_event=False):
        super(SaveButton, self).__init__(parent)
        if use_custom_event:
            self.parent = parent
        self.use = use_custom_event
        self.setFixedSize(40, 40)
        self.setToolTip('Click to save.')
        self.setIcon(QIcon(AddressConfig.get_icon_adr('save')))
        self.setStyleSheet("QPushButton { "
                           "border-style: solid; "
                           "border-width: 0px; "
                           "border-radius: 0px;"
                           "background-color: rgba(223, 223, 223, 0);}"
                           "QPushButton::focus{outline: none;}"
                           "QPushButton::hover {"
                           "border-style: solid;"
                           "border-width: 0px;"
                           "border-radius: 0px;"
                           "background-color: rgba(223, 223, 223, 150)}"
                           )

    def enterEvent(self, event):
        """
        Only serve for HoverLabel.
        """
        if self.use:
            if self.parent.func is not None:
                if self.parent.hover_win:
                    self.parent.hover_win.close()
                self.parent.hover_timer.stop()
                super().enterEvent(event)

    def leaveEvent(self, event):
        """
        Only serve for HoverLabel.
        """
        if self.use:
            if self.parent.func is not None:
                self.parent.hover_timer.start(1000)
                super().leaveEvent(event)


class SaveLabel(QLabel):
    """
    Overwrite QLabel to accept the QPixmap data from signal and then create it and also can save it.
    """

    def __init__(self, parent=None):
        super(SaveLabel, self).__init__(parent)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored))
        self.setStyleSheet("border: 1px solid #cccccc;")
        self.setScaledContents(True)

        self.save_btn = SaveButton(self)  # 若没有self，子控件的生命周期就不会由父控件管理，就可能不会按预期显示在界面上
        self.save_btn.clicked.connect(self.save_img)

        self.img = None

    def resizeEvent(self, event):
        self.save_btn.move(self.width() - self.save_btn.width(), 0)

    def update_img(self, img_data):
        pixmap = QPixmap()
        if pixmap.loadFromData(img_data, 'PNG'):
            self.img = pixmap
            self.setPixmap(pixmap)
        else:
            print("Failed to load image.")

    def save_img(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG File (*.png)")
        if self.img:
            if save_path:
                if self.img.save(save_path, 'PNG', -1):  # -1 使用默认质量
                    print("Pixmap saved successfully.")
                else:
                    print("Failed to save pixmap.")


class HoverLabel(SaveLabel):
    """
    Overwrite SaveLabel to use the HoverWindow.
    ---
    func = 0: show EEG image
    func = 1: show EEG canvas
    """

    def __init__(self, parent=None, size=None):
        super(HoverLabel, self).__init__(parent)
        self.hover_win = None
        self.raw = None
        self.func = None

        if size is not None:
            self.hover_win_size = QSize(size[0], size[1])
        else:
            self.hover_win_size = QSize(1400, 1000)

        self.hover_timer = QTimer(self)
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self.show_hover_window)

        self.save_btn.deleteLater()
        self.save_btn = SaveButton(self, True)
        self.save_btn.clicked.connect(self.save_img)

    def update_img(self, img_data=None, raw=None):
        super().update_img(img_data)

        if raw is None:
            self.func = 0
        else:
            self.func = 1
            self.raw = raw

    def enterEvent(self, event):
        if self.func is not None:
            self.hover_timer.start(1000)

    def leaveEvent(self, event):
        if self.func is not None:
            # 检查鼠标是否离开 HoverLabel
            if not self.rect().contains(self.mapFromGlobal(QCursor.pos())):
                # self.hover_win.hide()
                if self.hover_win:
                    self.hover_win.close()
                self.hover_timer.stop()
                super().leaveEvent(event)  # 调用基类的 leaveEvent 方法

    def show_hover_window(self):
        if self.func is not None:
            # 检查鼠标是否仍在 HoverLabel 内
            if self.rect().contains(self.mapFromGlobal(QCursor.pos())):
                if self.func == 0:
                    self.hover_win = HoverWindow(self)
                    # self.hover_win.show_image(self.hover_win_size)
                    # self.hover_win.show()
                    self.hover_win.show_image(self.img, self.hover_win_size)
                elif self.func == 1:
                    self.hover_win = HoverWindow(self, 1)
                    self.hover_win.show_eeg_raw(self.raw, self.hover_win_size)
                self.hover_win.show()


class HoverWindow(QWidget):
    """
    Overwrite QWidget to show the full EEG detected result generated by AD/SD.
    ---
    func = 0: show EEG image
    func = 1: show EEG canvas
    """

    def __init__(self, parent=None, func=0):
        super().__init__(parent, Qt.ToolTip)  # 初始化时被设置为 Qt.ToolTip 类型的窗口，这使得它的行为类似于工具提示——它没有边框
        self.parent = parent  # 若此处没有定义parent，则调用self.parent应视为方法，即self.parent().XXX

        if func == 0:  # img
            self.label = QLabel(self)
        else:  # raw
            self.layout = QVBoxLayout(self)
            # 去掉布局的间距和边距
            self.layout.setSpacing(0)  # 去掉组件之间的间隔
            self.layout.setContentsMargins(0, 0, 0, 0)  # 去掉布局的边缘间距

    def show_image(self, img, max_size):
        """
        显示图片悬浮窗
        """
        scaled_pixmap = img.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        self.label.adjustSize()  # 调整标签大小以适应图片
        self.adjustSize()  # 调整窗口大小以适应标签
        self.set_position(QCursor.pos())  # 设置位置再展示
        self.show()

    def show_eeg_raw(self, raw, max_size):
        """
        显示canvas悬浮窗
        """
        eeg_plot = raw.plot(duration=5, n_channels=len(raw.ch_names), scalings=300e-6, show=False)
        # 获取第一个轴对象，通常包含图表的标题和注释
        ax = eeg_plot.mne.ax_main
        # 修改纵轴刻度字体属性
        for label in ax.yaxis.get_ticklabels():
            if ThemeColorConfig.theme == 'dark':
                label.set_color('white')  # 字体颜色
            label.set_weight('bold')  # 字体加粗
            label.set_fontsize(11)  # 字体大小
        # 遍历这个轴上的所有文本对象，设置字体
        for text in ax.texts:
            text.set_weight('bold')
            text.set_fontsize(11)

        eeg_fig = FigureCanvas(eeg_plot)
        close()  # More than 20 figures have been opened.

        # 缩放绘图以适应 max_size
        eeg_fig.setFixedSize(max_size)

        eeg_fig.setFocusPolicy(Qt.StrongFocus)  # 将焦点策略设置为Qt.StrongFocus，以便接收键盘事件
        eeg_fig.mpl_connect('key_press_event', key_press_event)  # mpl_connect方法将键盘事件连接到canvas上
        self.layout.addWidget(eeg_fig)

        if raw.annotations:
            label = QLabel("SSW: Spike-slow wave", self)
            label.setAlignment(Qt.AlignCenter)  # 设置文本居中对齐
            label.setStyleSheet(f"""
                QLabel {{
                    border: 0px;
                    font-family: Arial;
                    font-size: 12pt;
                    color: {ThemeColorConfig.get_txt_color()};
                    font-weight: bold;
                    background-color: {rcParams['axes.facecolor']};
                    }}
                """)
            self.layout.addWidget(label)

        self.adjustSize()  # 调整窗口大小以适应标签
        self.set_position(QCursor.pos())  # 设置位置再展示
        self.show()

    def set_position(self, position):
        """
        将窗口的右下角放置在指定位置
        """
        # 获取当前鼠标所在屏幕的几何信息
        screen = QApplication.screenAt(position)
        if not screen:
            screen = QApplication.primaryScreen()

        screen_geometry = screen.geometry()

        # 获取窗口尺寸
        window_size = self.size()
        new_x = position.x() - window_size.width()
        new_y = position.y() - window_size.height()

        # 确保窗口不超出屏幕的顶部与左边
        if new_y < screen_geometry.top():
            new_y = screen_geometry.top()
        if new_x < screen_geometry.left():
            new_x = screen_geometry.left()

        # 移动窗口
        self.move(new_x, new_y)

    def leaveEvent(self, event):
        # self.hide()  # 隐藏窗口
        self.close()
        self.parent.hover_timer.stop()
        super().leaveEvent(event)  # 调用基类的 leaveEvent 方法


def key_press_event(event):
    """
    键盘事件处理函数
    """
    pressed_key = event.key
    print(pressed_key)


class MultiFuncEdit(QLineEdit):
    """
    Overwrite QLineEdit to use the mouse single/double click event.
    Default unable, need to be activated.
    ---
    mode = 0(S)/1(M): use for single/multi    default: 1
    func = 0(I)/1(O): use for input/output    default: 0
    ---
    single click:
        When SISO, select the input and opt the output file;
        When MIMO, select the input file and opt the output folder;
    double click:
        Open the current folder.
    """

    def __init__(self, parent=None, mode=1, func=0):
        super(MultiFuncEdit, self).__init__(parent)
        font = QFont("Microsoft YaHei", 11, QFont.Bold)
        self.setFont(font)

        if func == 0:  # 0 将被解释为假，而所有其他数字（无论是正数、负数还是非零浮点数）都将被解释为真。
            self.setPlaceholderText("— Click to Input —")
        self.func = func
        self.input_list = []
        self.set_mode_and_ini(mode)
        self.mode = mode

        self.click_timer = QTimer(self)
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.on_single_click)
        self.clicked_once = False
        self.double_click_handled = False

        self.setEnabled(False)

    def set_mode_and_ini(self, mode):
        self.input_list = []
        if self.func == 0:
            self.setText('')
        if self.func == 1:
            desktop_path = self.get_desktop_path()
            if desktop_path:
                if mode == 0:
                    output_path = path.join(desktop_path, 'output.mp4')
                    self.setText(output_path)
                if mode == 1:
                    self.setText(desktop_path)
            else:
                self.setPlaceholderText("— Click to Save —")

        self.mode = mode

    def mousePressEvent(self, event):
        if not self.clicked_once:
            self.clicked_once = True
            self.double_click_handled = False  # 重置双击处理标志
            self.click_timer.start(300)  # 300毫秒内没有第二次点击则视为单击
        else:
            self.clicked_once = False
            self.click_timer.stop()
            self.mouseDoubleClickEvent(event)

    def on_single_click(self):
        if self.double_click_handled:
            return  # 如果双击事件已处理，则忽略单击
        self.clicked_once = False

        if self.func == 0:
            if self.mode == 0:
                input_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                                            "Video File (*.mp4 *.avi *.rmvb)")
                if input_path:
                    self.input_list = [input_path]
                    input_path = path.basename(input_path)
                    self.setText(input_path)  # 设置文本为视频路径
            if self.mode == 1:
                input_paths, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "",
                                                              "Video Files (*.mp4 *.avi *.rmvb)")  # 单个文件夹中选择多个文件
                if input_paths:
                    self.input_list = input_paths
                    input_paths = [path.basename(p) for p in input_paths]
                    self.setText(';\n'.join(input_paths))  # 设置文本为视频路径

        if self.func == 1:
            if self.mode == 0:
                save_path, _ = QFileDialog.getSaveFileName(self, "Select Save Location", "", "Video File (*.mp4)")
                if save_path:
                    self.setText(save_path)
            if self.mode == 1:
                folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")  # 单个文件夹
                if folder:
                    self.setText(folder)

    def setText(self, text):
        super(MultiFuncEdit, self).setText(text)
        self.setToolTip(text)

    def mouseDoubleClickEvent(self, event):
        self.double_click_handled = True  # 设置双击事件已处理的标志
        self.clicked_once = False  # 重置单击标志
        self.click_timer.stop()  # 停止计时器

        if self.func == 0:
            if len(self.input_list) == 0:
                return
            else:
                adr = self.input_list[0]
                self.open_folder(adr)
        if self.func == 1:
            adr = self.text()
            self.open_folder(adr)  # 打开文件夹

            # elif path.isfile(adr):
            #     self.open_path(path.dirname(adr))  # 打开文件

    @staticmethod
    def open_folder(adr):
        # 文件夹
        normalized_path = path.normpath(adr)
        folder_path = normalized_path if path.isdir(normalized_path) else path.dirname(normalized_path)

        if path.exists(folder_path):
            try:
                if system() == "Windows":
                    startfile(folder_path)
                elif system() == "Darwin":
                    run(["open", folder_path])
                else:
                    run(["xdg-open", folder_path])
            except Exception as e:
                print(e)

    @staticmethod
    def open_path(adr):
        # 地址
        if platform == 'win32':
            startfile(adr)
        elif platform == 'darwin':
            Popen(['open', adr])
        else:  # Assume it's a Unix-like system
            Popen(['xdg-open', adr])

    @staticmethod
    def get_desktop_path():
        try:
            if system() == "Windows":  # Windows
                desktop_path = path.join(environ["USERPROFILE"], "Desktop")
            elif system() in ("Darwin", "Linux"):  # MacOS, Linux
                desktop_path = path.join(path.expanduser("~"), "Desktop")
            else:
                return None
            return desktop_path
        except Exception as e:
            print(e)
            return None


# class SignalDockArea(DockArea):
#     """
#     Overwrite DockArea to send the signal when TempArea closed.
#     """
#     restate_signal = pyqtSignal()
#
#     def removeTempArea(self, area):
#         super(SignalDockArea, self).removeTempArea(area)
#         self.restate_signal.emit()


class QWidgetDock(Dock):
    """
    Overwrite Dock to integrate QWidget.
    """

    def __init__(self, name, area, size=(10, 10), widget=None, hideTitle=True):
        super(QWidgetDock, self).__init__(name=name, area=area, size=size, widget=widget, hideTitle=hideTitle)

        # Transparent background
        self.setStyleSheet("background: transparent;")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

        # Initialize widget if provided
        self.widget = widget
        if self.widget:
            self.set_widget(self.widget)

    def set_widget(self, widget):
        """
        Dynamically replace the contained widget.
        """
        if self.widget and self.widget in self.layout.children():
            self.layout.removeWidget(self.widget)
            self.widget.setParent(None)
            self.widget.deleteLater()
        self.addWidget(widget)
        self.widget = widget


class SaveDock(Dock):
    """
    Overwrite Dock to integrate FigureCanvas already loaded its figure and save the contained figure.
    If @param fig_title is not None, adjust its h_pos in the figure during Dock resizing.
    If @param enable_scroll is True, use QScrollArea, be sure the canvas setSizePolicy and setFixedSize before load into.
    ---
    Methodology:
    1080: dock_height / screen_height: 0.22407407407407406 - 0.9342592592592592
    Establish a linear func to decline the h_pos: (0.3, 0.9) -> (0.9, 0.7) => y = -0.333x + 1
    """

    def __init__(self, name, area, canvas, size=(10, 10), hideTitle=False, fig_title=None, enable_scroll=False):
        super(SaveDock, self).__init__(name=name, area=area, size=size, hideTitle=hideTitle)
        self.canvas = canvas
        self.fig_title = fig_title

        if enable_scroll:
            # 使用带滚动条的控件容纳图像
            scroll_area = QScrollArea()
            scroll_area.setWidget(canvas)
            scroll_area.setWidgetResizable(True)
            self.addWidget(scroll_area)
        else:
            self.addWidget(canvas)

        self.save_btn = SaveButton(self)
        self.save_btn.clicked.connect(self.save_fig)

    def resizeEvent(self, event):
        self.save_btn.move(self.width() - self.save_btn.width(), 0)

        if self.fig_title:
            dock_height = self.size().height()
            screen_height = QApplication.desktop().screenGeometry().height()
            y = 0.97 - 0.12 * (dock_height / screen_height)
            y = min(max(y, 0.80), 0.97)  # y 一般落在 0.85 ~ 0.94 之间
            self.canvas.figure.suptitle(self.fig_title, fontsize=20, y=y)

        super(SaveDock, self).resizeEvent(event)

    def save_fig(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG File (*.png)")
        if save_path:
            self.canvas.figure.savefig(save_path, dpi=600, bbox_inches='tight')
