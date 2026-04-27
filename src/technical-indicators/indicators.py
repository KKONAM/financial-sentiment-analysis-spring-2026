
import pandas as pd
import numpy as np
from util import get_data

class indicators(object):
    def __init__(self, symbol = "JPM", dates=pd.bdate_range(start='2019-01-01', end='2022-12-31'), lookback_period=14):
        """
        Constructor method
        """
        self.symbol = symbol
        self.high, self.low, self.adj_close, self.volume = get_data(symbol = symbol, dates = dates)
        self.high = self.high.ffill().bfill()
        self.low = self.low.ffill().bfill()
        self.adj_close = self.adj_close.ffill().bfill()
        self.volume = self.volume.ffill().bfill()
        self.lookback_period = lookback_period
        pass

    def _get_ema(self, price_dataframe, span_size):
        # Found this syntax through investopedia
        ema = price_dataframe.ewm(span=span_size, adjust = False).mean()
        return ema

    def _get_sma(self, price_dataframe, window):
        sma = price_dataframe.rolling(window=window, min_periods=window).mean()
        return sma

    def get_cci_indicator(self):
        typical_price = (self.adj_close.copy() + self.high + self.low) / 3.0
        # I had to look up syntax for mean deviation
        mean_deviation = (
            typical_price
            .rolling(window=self.lookback_period, min_periods=self.lookback_period)
            .apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        )
        typical_price_sma = self._get_sma(typical_price, self.lookback_period)
        cci = (typical_price - typical_price_sma) / (0.015 * mean_deviation)

        return cci

    # Adapted from powerpoint presented in lecture
    def get_rsi_indicator(self):
        delta = self.adj_close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(self.lookback_period).mean()
        avg_loss = loss.rolling(self.lookback_period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # Adapted from powerpoint presented in lecture
    def get_bollinger_bands_indicator(self):
        prices = self.adj_close.copy()
        sma = self._get_sma(self.adj_close, self.lookback_period)
        rolling_std = prices.rolling(window=self.lookback_period, min_periods=self.lookback_period).std()
        top_band = sma + (rolling_std * 2)
        bottom_band = sma - (rolling_std * 2)
        self.sma_for_plot = sma
        self.bb_top_band_for_plot = top_band
        self.bb_bottom_band_for_plot = bottom_band
        bbp = (prices - bottom_band) / (top_band - bottom_band)
        return bbp

    def get_macd_indicator(self):
        fast_period = 12
        slow_period = 26
        signal_period = 9
        EMA_fast = self._get_ema(self.adj_close, span_size=fast_period)
        EMA_slow = self._get_ema(self.adj_close, span_size=slow_period)
        MACD = EMA_fast - EMA_slow
        Signal = self._get_ema(MACD, span_size=signal_period)
        self.macd_line_for_plot = MACD
        self.macd_signal_Line_for_plot = Signal

        MACD_norm = (MACD - MACD.rolling(self.lookback_period).mean()) / MACD.rolling(self.lookback_period).std()
        Signal_norm = (Signal - Signal.rolling(self.lookback_period).mean()) / Signal.rolling(self.lookback_period).std()

        Trade_Signal = MACD_norm - Signal_norm
        return Trade_Signal

    def get_momentum_indicator(self):
        momentum = (self.adj_close / self.adj_close.shift(self.lookback_period)) - 1
        return momentum
    
    def get_returns(self):
        return self.adj_close.pct_change()  