from __future__ import absolute_import
import numpy as np
from pyti.function_helper import fill_for_noncomputable_vals
from pyti.relative_strength_index import RSI
from pyti.relative_strength_index import relative_strength_index
from six.moves import range

class STOCHRSI(RSI):
    def __init__(self, data, period):
        super().__init__(data, period)
        self.stochrsi = stochrsi(data, period)

    def get_new_stochrsi(self, data, commit=False):
        range_rsi = None
        new_rsi = super().get_new_rsi(data, commit)
        range_rsi = np.concatenate((self.rsi[-self.period+1:], np.array([new_rsi])), axis=0)
        new_stochrsi = 100 * ((new_rsi - np.min(range_rsi)) / (
            np.max(range_rsi) - np.min(range_rsi)))
        return new_stochrsi

def stochrsi(data, period):
    """
    StochRSI.

    Formula:
    SRSI = ((RSIt - RSI LOW) / (RSI HIGH - LOW RSI)) * 100
    """
    rsi = relative_strength_index(data, period)
    stochrsi = [100 * ((rsi[idx] - np.min(rsi[idx + 1 - period:idx + 1])) / (
        np.max(rsi[idx + 1 - period:idx + 1]) - np.min(rsi[idx + 1 - period:idx + 1]))) for idx in
                range(period - 1, len(rsi))]
    stochrsi = fill_for_noncomputable_vals(data, stochrsi)
    return stochrsi