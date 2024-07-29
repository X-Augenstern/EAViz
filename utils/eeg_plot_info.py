from numpy import linspace

scale_factor_percent_list = list(linspace(0.1, 2.0, 20))  # 10% - 200% (+-10%)


class EEGPlotInfo:
    """ Data structure for holding information for eeg plotting. """

    def __init__(self, total_channels):
        self.info_changed = True
        self.total_channels = total_channels
        self.base_amp = 300
        self.scale_idx = 9
        self.amplitude = None
        self.window_size = 5
        self.n_channels = 10
        self.calc_cur_amp()

    def calc_cur_amp(self):
        # 300e-6 = 300*10^-6 = 300*10**-6  !=  300*10e-6 = 300*10*10^-6
        self.amplitude = self.base_amp * 10 ** -6 * scale_factor_percent_list[self.scale_idx]

    def plus_scale_idx(self):
        if self.scale_idx == len(scale_factor_percent_list) - 1:
            return
        self.scale_idx += 1
        self.calc_cur_amp()
        self.info_changed = True

    def minus_scale_idx(self):
        if self.scale_idx == 0:
            return
        self.scale_idx -= 1
        self.calc_cur_amp()
        self.info_changed = True

    def set_base_amp(self, val):
        if self.base_amp == val:
            return
        self.base_amp = val
        self.scale_idx = 9
        self.calc_cur_amp()
        self.info_changed = True

    def set_window_size(self, val):
        if self.window_size == val:
            return
        self.window_size = val
        self.info_changed = True

    def plus_n_channels(self):
        if self.n_channels == self.total_channels:
            return
        self.n_channels += 1
        self.info_changed = True

    def minus_n_channels(self):
        if self.n_channels == 1:
            return
        self.n_channels -= 1
        self.info_changed = True
