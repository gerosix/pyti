from __future__ import absolute_import
import numpy as np
from pyti import catch_errors
from pyti.function_helper import fill_for_noncomputable_vals
from six.moves import range
from six.moves import zip


class RSI():
    """
    compute a stream for the RSI indicator
    """
    def __init__(self, data=None, period=14):
        self.data = data
        self.period = int(period)
        self.avg_gain = 0
        self.avg_loss = 0
        self.rsi = []
        self.gains = []
        self.losses = []
        self.changes = []

        # compute realtive strength index
        if data is not None and period is not None:
            self.relative_strength_index(self.data, self.period)

    def get_new_rsi(self, data, commit=False):
        rsi = 0
        avg_loss = self.avg_loss
        avg_gain = self.avg_gain
        new_change = data - self.data[-1]
        if new_change > 0:
            new_gain = new_change
            avg_gain = ((self.avg_gain * (self.period - 1) +
                    new_gain) / self.period)
        else:
            new_loss = abs(new_change)
            avg_loss = ((self.avg_loss * (self.period - 1) +
                    new_loss) / self.period)

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        if commit is True:
            self.rsi.append(rsi)
        return rsi

    def relative_strength_index(self, data, period):
        """
        Relative Strength Index.

        Formula:
        RSI = 100 - (100 / 1 + (prevGain/prevLoss))
        """
        catch_errors.check_for_period_error(data, period)

        period = int(period)
        self.changes = [data_tup[1] - data_tup[0] for data_tup in zip(data[::1], data[1::1])]

        filtered_gain = [val < 0 for val in self.changes]
        self.gains = [0 if filtered_gain[idx] is True else self.changes[idx] for idx in range(0, len(filtered_gain))]

        filtered_loss = [val > 0 for val in self.changes]
        self.losses = [0 if filtered_loss[idx] is True else abs(self.changes[idx]) for idx in range(0, len(filtered_loss))]

        self.avg_gain = np.mean(self.gains[:period])
        self.avg_loss = np.mean(self.losses[:period])

        # first value
        rsi = []
        if self.avg_loss == 0:
            rsi.append(100)
        else:
            rs = self.avg_gain / self.avg_loss
            rsi.append(100 - (100 / (1 + rs)))

        # others
        for idx in range(1, len(data) - period):
            avg_gain = ((self.avg_gain * (period - 1) +
                        self.gains[idx + (period - 1)]) / period)
            avg_loss = ((self.avg_loss * (period - 1) +
                        self.losses[idx + (period - 1)]) / period)

            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))

        rsi = fill_for_noncomputable_vals(data, rsi)
        self.rsi = rsi
        return rsi

def relative_strength_index(data, period=14):
    """
    Relative Strength Index.

    Formula:
    RSI = 100 - (100 / 1 + (prevGain/prevLoss))
    """
    catch_errors.check_for_period_error(data, period)

    period = int(period)
    changes = [data_tup[1] - data_tup[0] for data_tup in zip(data[::1], data[1::1])]

    filtered_gain = [val < 0 for val in changes]
    gains = [0 if filtered_gain[idx] is True else changes[idx] for idx in range(0, len(filtered_gain))]

    filtered_loss = [val > 0 for val in changes]
    losses = [0 if filtered_loss[idx] is True else abs(changes[idx]) for idx in range(0, len(filtered_loss))]

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # first value
    rsi = []
    if avg_loss == 0:
        rsi.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))

    # others
    for idx in range(1, len(data) - period):
        avg_gain = ((avg_gain * (period - 1) +
                    gains[idx + (period - 1)]) / period)
        avg_loss = ((avg_loss * (period - 1) +
                    losses[idx + (period - 1)]) / period)

        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))

    rsi = fill_for_noncomputable_vals(data, rsi)
    return rsi[period:]