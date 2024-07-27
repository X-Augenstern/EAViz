class FilterInfo:
    """ Data structure for holding information for filtering. """

    def __init__(self):
        """ Constructor of the filter info object.

            Variables:
                hf - the upper cutoff frequency of the filter
                lf - the lower cutoff frequency of the filter
                notch - the notch frequency of the filter
                bp_lf - the lower cutoff frequency of the filter
                bp_hf - the upper cutoff frequency of the filter
                do_lp - whether to lowpass filter or not
                do_hp - whether to high pass filter or not
                do_notch - whether to notch filter or not
                do_bp - whether to bandpass filter or not
                filter_canceled - used to determine if filtering
                    was canceled during edf saving
        """
        self.hf = 70  # lowpass
        self.lf = 1  # highpass
        self.notch = 50
        self.bp_hf = 0
        self.bp_lf = 0
        self.do_lp = 1
        self.do_hp = 1
        self.do_notch = 1
        self.do_bp = 0
        self.filter_canceled = 0
