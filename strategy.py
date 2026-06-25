import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
import logging

logger = logging.getLogger(__name__)


def calculate_vwap(df, window=100):
    """Rolling Volume Weighted Average Price"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].rolling(window=window, min_periods=1).sum()
    cum_tp_vol = (typical_price * df['volume']).rolling(window=window, min_periods=1).sum()
    return cum_tp_vol / cum_vol


def calculate_supertrend(df, period=10, multiplier=3.0):
    """SuperTrend indikatörü: (trend_line, direction) döndürür.
    direction = 1 (yukari trend), -1 (asagi trend)"""
    atr = AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=period
    ).average_true_range()
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if pd.isna(atr.iloc[i]):
            supertrend.iloc[i] = np.nan
            direction.iloc[i] = 0
            continue

        if i == 0 or pd.isna(supertrend.iloc[i - 1]):
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            close_price = df['close'].iloc[i]
            prev_st = supertrend.iloc[i - 1]
            if close_price > prev_st:
                direction.iloc[i] = 1
            elif close_price < prev_st:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], prev_st)
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], prev_st)

    return supertrend, direction


class MultiIndicatorStrategy:
    def __init__(self,
                 atr_period=14,
                 volume_sma_period=20,
                 volume_threshold=0.7,
                 use_trend_filter=False,
                 enable_short=False,
                 vwap_window=100,
                 ema_trend_period=200,
                 super_trend_period=10,
                 super_trend_multiplier=2.0,
                 adx_threshold=25.0):
        self.atr_period = atr_period
        self.volume_sma_period = volume_sma_period
        self.volume_threshold = volume_threshold
        self.use_trend_filter = use_trend_filter
        self.enable_short = enable_short
        self.vwap_window = vwap_window
        self.ema_trend_period = ema_trend_period  # 1h EMA 200 (5m icin 2400 esdegeri)
        self.super_trend_period = super_trend_period
        self.super_trend_multiplier = super_trend_multiplier
        self.adx_threshold = adx_threshold

    def _calculate_indicators(self, df: pd.DataFrame):
        min_len = max(self.atr_period, self.vwap_window, self.ema_trend_period) + 10
        if len(df) < min_len:
            return None, None, None, None, None, None

        atr = AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=self.atr_period
        )
        vwap = calculate_vwap(df, window=self.vwap_window)
        ema_trend = EMAIndicator(close=df['close'], window=self.ema_trend_period)
        supertrend, st_direction = calculate_supertrend(
            df, period=self.super_trend_period, multiplier=self.super_trend_multiplier
        )
        adx = ADXIndicator(
            high=df['high'], low=df['low'], close=df['close'], window=self.atr_period
        )

        return (
            atr.average_true_range(),
            vwap,
            ema_trend.ema_indicator(),
            supertrend,
            st_direction,
            adx.adx(),
        )

    def get_1h_trend(self, df_1h: pd.DataFrame):
        """1 saatlik grafikte EMA200 trend yönünü döndür: 1=yukari, -1=asagi, 0=belirsiz"""
        if df_1h is None or df_1h.empty or len(df_1h) < self.ema_trend_period + 5:
            return 0
        ema_trend = EMAIndicator(close=df_1h['close'], window=self.ema_trend_period)
        ema_series = ema_trend.ema_indicator()
        if pd.isna(ema_series.iloc[-1]):
            return 0
        if df_1h['close'].iloc[-1] > ema_series.iloc[-1]:
            return 1
        elif df_1h['close'].iloc[-1] < ema_series.iloc[-1]:
            return -1
        return 0

    def get_signal(self, df: pd.DataFrame, df_1h: pd.DataFrame = None) -> str:
        atr_series, vwap_series, ema_trend_series, st_series, st_dir, adx_series = self._calculate_indicators(df)
        if atr_series is None:
            return "HOLD"

        current_close = df['close'].iloc[-1]
        current_atr = atr_series.iloc[-1]
        current_vwap = vwap_series.iloc[-1]
        current_ema_trend = ema_trend_series.iloc[-1]
        current_st = st_series.iloc[-1]
        current_st_dir = st_dir.iloc[-1]
        current_adx = adx_series.iloc[-1]

        if (pd.isna(current_atr) or pd.isna(current_vwap)
                or pd.isna(current_ema_trend) or pd.isna(current_st)
                or pd.isna(current_adx)):
            return "HOLD"

        # 1 saatlik trend onayi
        trend_1h = self.get_1h_trend(df_1h)
        if trend_1h == 0 and df_1h is not None:
            return "HOLD"

        logger.debug(
            f"Close: {current_close:.2f} | EMA200: {current_ema_trend:.2f} | "
            f"VWAP: {current_vwap:.2f} | ST_dir: {current_st_dir} | ADX: {current_adx:.1f}"
        )

        # ADX trend gücü filtresi: zayıf trendte işlem yapma
        if current_adx < self.adx_threshold:
            logger.debug(f"ADX zayıf ({current_adx:.1f} < {self.adx_threshold}), HOLD")
            return "HOLD"

        # Hacim onayi
        volume_confirmed = False
        if 'volume' in df.columns and len(df) >= self.volume_sma_period:
            volume_sma = df['volume'].rolling(
                window=self.volume_sma_period
            ).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            if pd.notna(volume_sma) and volume_sma > 0:
                volume_confirmed = current_volume >= (volume_sma * self.volume_threshold)

        # 5m Trend: EMA 200
        above_ema200_5m = current_close > current_ema_trend
        below_ema200_5m = current_close < current_ema_trend

        # VWAP durum
        above_vwap = current_close > current_vwap
        below_vwap = current_close < current_vwap

        # SuperTrend sinyali: kesişim onayi
        prev_st_dir = st_dir.iloc[-2] if len(st_dir) > 1 else current_st_dir
        st_cross_up = (prev_st_dir == -1 and current_st_dir == 1)
        st_cross_down = (prev_st_dir == 1 and current_st_dir == -1)
        st_buy_signal = st_cross_up
        st_sell_signal = st_cross_down

        signal = "HOLD"

        # Long: 1h yukari trend + 5m EMA200 ustu + VWAP ustu + SuperTrend AL kesişimi + hacim onayi + ADX onayi
        if (trend_1h == 1 and above_ema200_5m and above_vwap
                and st_buy_signal and volume_confirmed and current_adx >= self.adx_threshold):
            signal = "BUY"
        # Short: 1h asagi trend + 5m EMA200 alti + VWAP alti + SuperTrend SAT kesişimi + hacim onayi + ADX onayi
        elif (trend_1h == -1 and below_ema200_5m and below_vwap
                  and st_sell_signal and volume_confirmed and current_adx >= self.adx_threshold):
            if self.enable_short:
                signal = "SHORT"
            else:
                signal = "SELL"

        return signal

    def get_super_trend_direction(self, df: pd.DataFrame):
        """Mevcut SuperTrend yönünü döndür: 1=yukari, -1=asagi, 0=belirsiz"""
        _, _, _, _, st_dir, _ = self._calculate_indicators(df)
        if st_dir is None:
            return 0
        current_st_dir = st_dir.iloc[-1]
        if pd.isna(current_st_dir):
            return 0
        return int(current_st_dir)

    def get_last_atr(self, df: pd.DataFrame):
        atr_series, _, _, _, _, _ = self._calculate_indicators(df)
        if atr_series is None:
            return None
        return atr_series.iloc[-1]
