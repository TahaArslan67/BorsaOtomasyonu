"""
GMSTR Özellik Mühendisliği Modülü
- 150+ teknik indikatör
- Makro özellikler (benchmark, alpha, beta, korelasyon)
- Target redesign (3-sınıf ve dinamik eşik)
"""
import numpy as np
import pandas as pd
import ta
from typing import List, Literal
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureEngineer:
    """Kapsamlı teknik analiz + makro özellik üretici."""

    def __init__(self):
        self.feature_cols: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm özellikleri hesapla."""
        df = df.copy()
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
        o = df['Open']

        # Geleceğe dönük leakage kolonlarını önceden temizle (varsa)
        leakage_prefixes = ('target_', 'future_ret_', 'threshold_dyn_', 'target_dyn_', 'target_3c_', 'target_zscore_')
        drop_cols = [c for c in df.columns if c.startswith(leakage_prefixes)]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # ================================================================
        # 1. MOMENTUM INDICATORS
        # ================================================================
        for period in [5, 7, 9, 14, 21]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(c, period).rsi()

        stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_kd_diff'] = df['stoch_k'] - df['stoch_d']

        df['williams_r'] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
        df['tsi'] = ta.momentum.TSIIndicator(c).tsi()

        for period in [5, 10, 14]:
            df[f'roc_{period}'] = ta.momentum.ROCIndicator(c, period).roc()

        # ================================================================
        # 2. TREND INDICATORS
        # ================================================================
        for fast, slow, signal in [(12, 26, 9), (5, 13, 5), (8, 17, 9)]:
            tag = f'_{fast}_{slow}'
            macd = ta.trend.MACD(c, slow, fast, signal)
            df[f'macd{tag}'] = macd.macd()
            df[f'macd_sig{tag}'] = macd.macd_signal()
            df[f'macd_diff{tag}'] = macd.macd_diff()

        for w in [5, 8, 13, 21, 34, 55, 89]:
            df[f'ema_{w}'] = ta.trend.EMAIndicator(c, w).ema_indicator()

        for w in [10, 20, 50, 100]:
            df[f'sma_{w}'] = ta.trend.SMAIndicator(c, w).sma_indicator()

        adx = ta.trend.ADXIndicator(h, l, c, 14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        df['adx_diff'] = df['adx_pos'] - df['adx_neg']

        for w in [14, 20]:
            df[f'cci_{w}'] = ta.trend.CCIIndicator(h, l, c, w).cci()

        df['dpo'] = ta.trend.DPOIndicator(c, 20).dpo()

        # Ichimoku
        ich = ta.trend.IchimokuIndicator(h, l, 9, 26, 52)
        df['ich_a'] = ich.ichimoku_a()
        df['ich_b'] = ich.ichimoku_b()
        df['ich_base'] = ich.ichimoku_base_line()

        aroon = ta.trend.AroonIndicator(h, l, 14)
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_ind'] = aroon.aroon_indicator()

        # ================================================================
        # 3. VOLATILITY INDICATORS
        # ================================================================
        for std in [1.5, 2.0, 2.5]:
            tag = str(std).replace('.', '')
            bb = ta.volatility.BollingerBands(c, 20, std)
            df[f'bb_pct_{tag}'] = bb.bollinger_pband()
            df[f'bb_w_{tag}'] = bb.bollinger_wband()

        for w in [7, 14]:
            df[f'atr_{w}'] = ta.volatility.AverageTrueRange(h, l, c, w).average_true_range()
        df['natr'] = df['atr_14'] / c.replace(0, np.nan)

        kc = ta.volatility.KeltnerChannel(h, l, c)
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_low'] = kc.keltner_channel_lband()

        dc = ta.volatility.DonchianChannel(h, l, c)
        df['dc_high'] = dc.donchian_channel_hband()
        df['dc_low'] = dc.donchian_channel_lband()

        # ================================================================
        # 4. VOLUME INDICATORS
        # ================================================================
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v, 14).volume_weighted_average_price()
        df['mfi'] = ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index()
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow()
        df['fi'] = ta.volume.ForceIndexIndicator(c, v, 13).force_index()

        eom_raw = ta.volume.EaseOfMovementIndicator(h, l, v, 14).ease_of_movement()
        df['eom'] = eom_raw.replace([np.inf, -np.inf], np.nan)

        df['adi'] = ta.volume.AccDistIndexIndicator(h, l, c, v).acc_dist_index()

        df['vol_ema20'] = v.ewm(span=20).mean()
        df['vol_ratio'] = v / df['vol_ema20'].replace(0, np.nan)
        df['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
        df['vol_surge'] = (df['vol_ratio'] > 2.0).astype(int)

        # ================================================================
        # 5. RETURN & PRICE DERIVATIVES
        # ================================================================
        for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
            df[f'ret_{lag}d'] = c.pct_change(lag)

        # Trend eğimi
        for window in [5, 10, 20]:
            x = np.arange(window)
            def linreg_slope(prices):
                if len(prices) < window:
                    return np.nan
                return np.polyfit(x, prices.values, 1)[0]
            df[f'trend_slope_{window}'] = c.rolling(window).apply(linreg_slope, raw=False)
            df[f'trend_slope_{window}_norm'] = df[f'trend_slope_{window}'] / c

        df['price_accel'] = c.diff().diff()
        df['log_ret'] = np.log(c / c.shift(1))

        for w in [5, 10, 20, 40]:
            df[f'rv_{w}'] = df['log_ret'].rolling(w).std() * np.sqrt(252)

        df['vol_ratio_5_20'] = df['rv_5'] / df['rv_20'].replace(0, np.nan)
        df['vol_regime'] = (df['rv_5'] > df['rv_20']).astype(int)

        # ================================================================
        # 6. PRICE POSITION & CROSSOVERS
        # ================================================================
        for w in [5, 8, 13, 21, 34, 55]:
            df[f'pp_ema_{w}'] = (c - df[f'ema_{w}']) / df[f'ema_{w}'].replace(0, np.nan)

        pairs = [(5, 8), (8, 13), (13, 21), (21, 34), (34, 55)]
        for fast, slow in pairs:
            df[f'ema_cross_{fast}_{slow}'] = (df[f'ema_{fast}'] > df[f'ema_{slow}']).astype(int)

        ema_cols = [f'ema_cross_{a}_{b}' for a, b in pairs]
        df['ema_score'] = df[ema_cols].sum(axis=1)

        df['golden_cross'] = (df['sma_50'] > df['sma_100'].shift(1)) & (df['sma_50'] <= df['sma_100'].shift(2))
        df['death_cross'] = (df['sma_50'] < df['sma_100'].shift(1)) & (df['sma_50'] >= df['sma_100'].shift(2))

        # ================================================================
        # 7. CANDLESTICK PATTERNS
        # ================================================================
        df['body'] = (c - o) / c.replace(0, np.nan)
        df['upper_wick'] = (h - c.clip(lower=o)) / c.replace(0, np.nan)
        df['lower_wick'] = (c.clip(upper=o) - l) / c.replace(0, np.nan)
        df['hl_range'] = (h - l) / c.replace(0, np.nan)
        df['hl_pos'] = (c - l) / (h - l + 1e-8)
        df['is_doji'] = (abs(df['body']) < 0.001).astype(int)

        # ================================================================
        # 8. SUPPORT/RESISTANCE PROXIMITY
        # ================================================================
        for w in [10, 20, 40]:
            df[f'dist_high_{w}'] = (h.rolling(w).max() - c) / c.replace(0, np.nan)
            df[f'dist_low_{w}'] = (c - l.rolling(w).min()) / c.replace(0, np.nan)

        # ================================================================
        # 9. RSI DIVERGENCE
        # ================================================================
        df['price_mom_5'] = c.pct_change(5)
        df['rsi_14_mom_5'] = df['rsi_14'].diff(5)
        df['rsi_divergence'] = df['rsi_14_mom_5'] * np.sign(df['price_mom_5'])

        # ================================================================
        # 10. COMPOSITE SCORES
        # ================================================================
        rsi_std = df['rsi_14'].rolling(40).std().replace(0, np.nan)
        df['rsi_z'] = (df['rsi_14'] - df['rsi_14'].rolling(40).mean()) / rsi_std

        df['mom_score'] = (
            np.sign(df['macd_diff_12_26']) * 0.20 +
            np.sign(df['pp_ema_5']) * 0.15 +
            np.sign(df['pp_ema_13']) * 0.10 +
            ((df['rsi_14'] - 50) / 50).clip(-1, 1) * 0.15 +
            ((df['stoch_k'] - 50) / 50).clip(-1, 1) * 0.10 +
            (df['cci_20'] / 200).clip(-1, 1) * 0.10 +
            (df['adx_diff'] / 100).clip(-1, 1) * 0.10 +
            np.sign(df['obv'].diff(5)).fillna(0) * 0.10
        )

        df['trend_strength'] = (
            df['adx'] / 100 * 0.4 +
            df['ema_score'] / 5 * 0.3 +
            (df['pp_ema_21'].abs() * 5).clip(0, 1) * 0.3
        )

        # ================================================================
        # 11. CYCLICAL TIME FEATURES
        # ================================================================
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        df['dom_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df.index.day / 31)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['qtr_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
        df['qtr_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)

        # ================================================================
        # 12. LAGGED FEATURES
        # ================================================================
        key_features = ['rsi_14', 'macd_diff_12_26', 'bb_pct_20', 'atr_14', 'mom_score']
        for feat in key_features:
            if feat in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

        # ================================================================
        # 13. MACRO FEATURES (Benchmark / Silver / Alpha)
        # ================================================================
        if 'Benchmark_Return' in df.columns and 'Fund_Return' in df.columns:
            bench = df['Benchmark_Return']
            fund = df['Fund_Return']

            # Benchmark günlük değişim (kümülatiften türet)
            bench_chg = bench.diff()
            fund_chg = fund.diff()

            df['macro_bench_ret_1d'] = bench_chg
            df['macro_bench_ret_5d'] = bench.diff(5)
            df['macro_bench_ret_10d'] = bench.diff(10)
            df['macro_bench_ret_20d'] = bench.diff(20)

            df['macro_bench_ma5'] = bench.rolling(5).mean()
            df['macro_bench_ma20'] = bench.rolling(20).mean()

            # Benchmark trend eğimi
            x5 = np.arange(5)
            def bench_slope(prices):
                if len(prices) < 5:
                    return np.nan
                return np.polyfit(x5, prices.values, 1)[0]
            df['macro_bench_trend'] = bench.rolling(5).apply(bench_slope, raw=False)

            # Alpha (fund - benchmark)
            df['macro_alpha_1d'] = fund_chg - bench_chg
            df['macro_alpha_5d'] = fund.diff(5) - bench.diff(5)
            df['macro_alpha_20d'] = fund.diff(20) - bench.diff(20)

            # Beta (rolling 20d: cov(fund, bench) / var(bench))
            cov = fund_chg.rolling(20).cov(bench_chg)
            var = bench_chg.rolling(20).var()
            df['macro_beta_20'] = cov / var.replace(0, np.nan)

            # Korelasyon
            df['macro_corr_20'] = fund_chg.rolling(20).corr(bench_chg)

            # Göreceli güç (fund / benchmark)
            df['macro_rel_strength'] = (fund + 100) / (bench + 100).replace(0, np.nan)

            # Benchmark volatilitesi
            df['macro_bench_vol_20'] = bench_chg.rolling(20).std()
            df['macro_fund_vol_20'] = fund_chg.rolling(20).std()

            # Regime: benchmark fonu aşıyor mu?
            df['macro_bench_lead'] = (bench_chg > fund_chg).astype(int)

            # Momentum farkı
            df['macro_mom_spread'] = (
                fund.diff(5).rolling(5).mean() - bench.diff(5).rolling(5).mean()
            )

        # ================================================================
        # TEMİZLİK - NaN doldurma
        # ================================================================
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill(limit=10)

        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)

        # Feature kolonlarını kaydet - leakage kolonlarını hariç tut
        exclude = {'Open', 'High', 'Low', 'Close', 'Volume',
                   'Fund_Return', 'Benchmark_Return'}
        leakage_prefixes = ('target_', 'future_ret_', 'threshold_dyn_', 'target_dyn_', 'target_3c_', 'target_zscore_')
        self.feature_cols = [c for c in df.columns
                             if c not in exclude
                             and not c.startswith(leakage_prefixes)]

        return df

    def get_feature_columns(self, df: pd.DataFrame, target_prefix: str = 'target') -> List[str]:
        exclude = {'Open', 'High', 'Low', 'Close', 'Volume',
                   'Fund_Return', 'Benchmark_Return'}
        leakage_prefixes = ('target_', 'future_ret_', 'threshold_dyn_', 'target_dyn_', 'target_3c_', 'target_zscore_')
        return [c for c in df.columns
                if c not in exclude
                and not c.startswith(leakage_prefixes)]

    @staticmethod
    def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 30) -> List[str]:
        """En bilgilendirici k feature'i SelectKBest ile secer."""
        # NaN temizligi
        mask = X.notna().all(axis=1) & y.notna()
        Xc = X[mask]
        yc = y[mask]
        if len(Xc) < k * 2:
            return list(X.columns)
        selector = SelectKBest(mutual_info_classif, k=min(k, len(X.columns)))
        selector.fit(Xc, yc)
        selected = X.columns[selector.get_support()].tolist()
        return selected


def create_targets(df: pd.DataFrame, horizons: dict, threshold: float = 0.0) -> pd.DataFrame:
    """Klasik binary target (threshold > 0 = yukarı)."""
    df = df.copy()
    for name, bars in horizons.items():
        future_ret = df['Close'].pct_change(bars).shift(-bars)
        df[f'target_{name}'] = (future_ret > threshold).astype(int)
        df[f'future_ret_{name}'] = future_ret
    return df


def create_targets_dynamic(df: pd.DataFrame, horizons: dict,
                           vol_window: int = 20,
                           multiplier: float = 0.5) -> pd.DataFrame:
    """
    Dinamik eşik: her noktada son `vol_window` günlük volatiliteye göre
    threshold ayarlanır. Düşük volatilite = daha düşük eşik, yüksek vol = yüksek eşik.
    """
    df = df.copy()
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    for name, bars in horizons.items():
        future_ret = df['Close'].pct_change(bars).shift(-bars)
        # Yıllıklandırılmış volatiliteyi bar sayısına göre ölçekle
        scaled_vol = vol * np.sqrt(bars / 252)
        dynamic_thresh = scaled_vol * multiplier
        df[f'target_dyn_{name}'] = (future_ret > dynamic_thresh).astype(int)
        df[f'future_ret_{name}'] = future_ret
        df[f'threshold_dyn_{name}'] = dynamic_thresh
    return df


def create_targets_3class(df: pd.DataFrame, horizons: dict,
                          z_thresh: float = 0.5) -> pd.DataFrame:
    """
    3-sınıf hedef:
      2 = Strong Up   (getiri > +z_thresh * std)
      1 = Neutral     (orta bant)
      0 = Strong Down (getiri < -z_thresh * std)
    Eğitimde 'neutral' sınıfları opsiyonel olarak filtreleyebiliriz.
    """
    df = df.copy()
    for name, bars in horizons.items():
        future_ret = df['Close'].pct_change(bars).shift(-bars)
        # Normalize by rolling volatility
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        rolling_std = log_ret.rolling(60).std() * np.sqrt(bars)
        z_score = future_ret / rolling_std.replace(0, np.nan)

        conditions = [
            z_score > z_thresh,
            z_score < -z_thresh,
        ]
        choices = [2, 0]
        df[f'target_3c_{name}'] = np.select(conditions, choices, default=1)
        df[f'future_ret_{name}'] = future_ret
        df[f'target_zscore_{name}'] = z_score
    return df


def filter_neutral(df: pd.DataFrame, horizons: List[str],
                   target_prefix: str = 'target_3c_') -> pd.DataFrame:
    """3-sınıf hedeflerde neutral (1) sınıfını filtrele (opsiyonel)."""
    mask = pd.Series(True, index=df.index)
    for h in horizons:
        col = f'{target_prefix}{h}'
        if col in df.columns:
            mask &= (df[col] != 1)
    return df[mask]
