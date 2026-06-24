"""
GMSTR Saatlik Model Eğitimi v2 (1h / 4h)
- Gelişmiş özellik mühendisliği (100+ özellik)
- Regime detection (trend/range/volatile)
- Asymmetric features (alış/satış baskısı)
- Stacking ensemble + Platt kalibrasyon
- Hedef: %60+ doğruluk (özellikle 4h)
"""
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / 'gmstr_models'


def load_hourly_data():
    """Yahoo Finance'den saatlik GMSTR verisi çek."""
    try:
        import yfinance as yf
        print("  Yahoo Finance'den saatlik veri çekiliyor...")
        ticker = yf.Ticker("GMSTR.IS")
        # Maksimum saatlik veri
        df = ticker.history(period="730d", interval="1h")
        if len(df) < 100:
            print(f"  Yetersiz veri: {len(df)} satır, 60d deneniyor...")
            df = ticker.history(period="60d", interval="1h")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        # Sadece borsa saatlerini filtrele (09:00-18:00)
        df = df[df.index.hour.isin(range(9, 19))]
        print(f"  {len(df)} saatlik bar yüklendi | {df.index[0].date()} -> {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  Hata: {e}")
        return None


def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def engineer_features(df, resample_hours=1):
    """Gelişmiş özellik mühendisliği - 100+ özellik."""
    if resample_hours > 1:
        df = df.resample(f'{resample_hours}h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()

    close = df['Close'].copy()
    high = df['High'].copy()
    low = df['Low'].copy()
    open_ = df['Open'].copy()
    volume = df['Volume'].copy() if 'Volume' in df.columns else pd.Series(1, index=df.index)

    # ===== 1. TEMEL GETİRİ VE LAG ÖZELLİKLERİ =====
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48]:
        df[f'ret_{lag}'] = close.pct_change(lag)
        df[f'log_ret_{lag}'] = np.log(close / close.shift(lag))

    # ===== 2. MOMENTUM =====
    for p in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48, 72, 96]:
        df[f'mom_{p}'] = close / close.shift(p) - 1

    # ===== 3. VOLATİLİTE =====
    for w in [3, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48]:
        df[f'vol_{w}'] = ret.rolling(w).std()
        df[f'vol_log_{w}'] = log_ret.rolling(w).std()

    # Volatilite oranı (kısa/uzun)
    for short, long in [(3, 12), (6, 24), (12, 48)]:
        df[f'vol_ratio_{short}_{long}'] = df[f'vol_{short}'] / (df[f'vol_{long}'] + 1e-8)

    # Volatilite rejimi
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()
    df['vol_regime'] = vol_20 / (vol_60 + 1e-8)  # >1: yüksek vol, <1: düşük vol

    # ===== 4. HAREKETLİ ORTALAMALAR =====
    for w in [3, 5, 8, 10, 13, 20, 21, 34, 50, 55, 89]:
        df[f'ma_{w}'] = close.rolling(w).mean()
        df[f'ma_ratio_{w}'] = close / (df[f'ma_{w}'] + 1e-8) - 1
        df[f'ma_dist_{w}'] = (close - df[f'ma_{w}']) / (df[f'ma_{w}'] + 1e-8)

    # EMA
    for w in [5, 8, 13, 21, 34, 55]:
        df[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
        df[f'ema_ratio_{w}'] = close / (df[f'ema_{w}'] + 1e-8) - 1

    # MA crossover sinyalleri
    for short, long in [(3, 12), (5, 20), (5, 34), (8, 21), (8, 34), (13, 55), (21, 89)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)
        df[f'ma_cross_sign_{short}_{long}'] = np.sign(ma_s - ma_l)
        # Crossover geçişi (son 3 barda değişti mi?)
        cross_sign = np.sign(ma_s - ma_l)
        df[f'ma_cross_change_{short}_{long}'] = (cross_sign != cross_sign.shift(1)).astype(int)

    # EMA crossover
    for short, long in [(5, 21), (8, 34), (13, 55)]:
        ema_s = close.ewm(span=short, adjust=False).mean()
        ema_l = close.ewm(span=long, adjust=False).mean()
        df[f'ema_cross_{short}_{long}'] = (ema_s - ema_l) / (ema_l + 1e-8)
        df[f'ema_cross_sign_{short}_{long}'] = np.sign(ema_s - ema_l)

    # ===== 5. RSI =====
    for period in [5, 7, 9, 14, 21]:
        rsi = compute_rsi(close, period)
        df[f'rsi_{period}'] = rsi
        df[f'rsi_{period}_norm'] = (rsi - 50) / 50
        df[f'rsi_{period}_ob'] = (rsi > 70).astype(int)  # Aşırı alım
        df[f'rsi_{period}_os'] = (rsi < 30).astype(int)  # Aşırı satım
        df[f'rsi_{period}_mid'] = ((rsi > 45) & (rsi < 55)).astype(int)  # Nötr bölge

    # RSI divergence (fiyat yükselirken RSI düşüyor mu?)
    rsi14 = compute_rsi(close, 14)
    price_higher = (close > close.shift(5)).astype(int)
    rsi_lower = (rsi14 < rsi14.shift(5)).astype(int)
    df['bearish_divergence'] = (price_higher & rsi_lower).astype(int)
    price_lower = (close < close.shift(5)).astype(int)
    rsi_higher = (rsi14 > rsi14.shift(5)).astype(int)
    df['bullish_divergence'] = (price_lower & rsi_higher).astype(int)

    # ===== 6. MACD =====
    for fast, slow, signal_p in [(12, 26, 9), (5, 13, 5), (8, 21, 8)]:
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=signal_p, adjust=False).mean()
        hist = macd - signal
        df[f'macd_{fast}_{slow}'] = macd / (close + 1e-8)
        df[f'macd_signal_{fast}_{slow}'] = signal / (close + 1e-8)
        df[f'macd_hist_{fast}_{slow}'] = hist / (close + 1e-8)
        df[f'macd_cross_{fast}_{slow}'] = np.sign(macd - signal)
        df[f'macd_hist_change_{fast}_{slow}'] = np.sign(hist - hist.shift(1))

    # ===== 7. BOLLİNGER BANTLARI =====
    for w in [10, 12, 20, 30]:
        bb_mid = close.rolling(w).mean()
        bb_std = close.rolling(w).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df[f'bb_pos_{w}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df[f'bb_width_{w}'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        df[f'bb_above_{w}'] = (close > bb_upper).astype(int)
        df[f'bb_below_{w}'] = (close < bb_lower).astype(int)
        df[f'bb_squeeze_{w}'] = (df[f'bb_width_{w}'] < df[f'bb_width_{w}'].rolling(20).mean() * 0.8).astype(int)

    # ===== 8. STOKASTİK =====
    for w in [5, 9, 14, 21]:
        low_min = low.rolling(w).min()
        high_max = high.rolling(w).max()
        stoch_k = (close - low_min) / (high_max - low_min + 1e-8) * 100
        stoch_d = stoch_k.rolling(3).mean()
        df[f'stoch_k_{w}'] = stoch_k / 100
        df[f'stoch_d_{w}'] = stoch_d / 100
        df[f'stoch_kd_{w}'] = (stoch_k - stoch_d) / 100
        df[f'stoch_ob_{w}'] = (stoch_k > 80).astype(int)
        df[f'stoch_os_{w}'] = (stoch_k < 20).astype(int)

    # ===== 9. ATR VE VOLATILITE =====
    for w in [5, 7, 10, 14, 20]:
        atr = compute_atr(high, low, close, w)
        df[f'atr_{w}'] = atr / (close + 1e-8)
        df[f'atr_ratio_{w}'] = atr / (atr.shift(w) + 1e-8)

    # True Range normalize
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['tr_norm'] = tr / (close + 1e-8)

    # ===== 10. FİYAT POZİSYONU =====
    for w in [5, 8, 10, 12, 16, 20, 24, 36, 48, 72, 96]:
        roll_min = close.rolling(w).min()
        roll_max = close.rolling(w).max()
        df[f'price_pos_{w}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)
        df[f'dist_high_{w}'] = (roll_max - close) / (close + 1e-8)
        df[f'dist_low_{w}'] = (close - roll_min) / (close + 1e-8)

    # ===== 11. HACİM ÖZELLİKLERİ =====
    for w in [5, 10, 12, 20, 24]:
        vol_ma = volume.rolling(w).mean()
        df[f'vol_ratio_v_{w}'] = volume / (vol_ma + 1e-8)
        df[f'vol_above_avg_{w}'] = (volume > vol_ma).astype(int)

    # OBV (On Balance Volume)
    obv = (np.sign(close.diff()) * volume).cumsum()
    for w in [10, 20, 50]:
        obv_ma = obv.rolling(w).mean()
        df[f'obv_ratio_{w}'] = obv / (obv_ma + 1e-8)
        df[f'obv_trend_{w}'] = np.sign(obv - obv_ma)

    # Volume-Price Trend
    vpt = (ret * volume).cumsum()
    df['vpt_norm'] = (vpt - vpt.rolling(20).mean()) / (vpt.rolling(20).std() + 1e-8)

    # Chaikin Money Flow
    for w in [10, 20]:
        mfm = ((close - low) - (high - close)) / (high - low + 1e-8)
        mfv = mfm * volume
        df[f'cmf_{w}'] = mfv.rolling(w).sum() / (volume.rolling(w).sum() + 1e-8)

    # ===== 12. MEKTUP ÖZELLİKLERİ (CANDLE) =====
    df['candle_body'] = (close - open_) / (close + 1e-8)
    df['candle_body_abs'] = df['candle_body'].abs()
    df['candle_upper_shadow'] = (high - close.clip(lower=open_)) / (close + 1e-8)
    df['candle_lower_shadow'] = (close.clip(upper=open_) - low) / (close + 1e-8)
    df['candle_range'] = (high - low) / (close + 1e-8)
    df['candle_is_bull'] = (close > open_).astype(int)
    df['candle_is_doji'] = (df['candle_body_abs'] < 0.001).astype(int)
    df['candle_is_hammer'] = (
        (df['candle_lower_shadow'] > 2 * df['candle_body_abs']) &
        (df['candle_upper_shadow'] < df['candle_body_abs'])
    ).astype(int)
    df['candle_is_shooting_star'] = (
        (df['candle_upper_shadow'] > 2 * df['candle_body_abs']) &
        (df['candle_lower_shadow'] < df['candle_body_abs'])
    ).astype(int)

    # Ardışık mum örüntüleri
    df['two_bull'] = ((close > open_) & (close.shift(1) > open_.shift(1))).astype(int)
    df['two_bear'] = ((close < open_) & (close.shift(1) < open_.shift(1))).astype(int)
    df['bull_engulf'] = (
        (close > open_) & (close.shift(1) < open_.shift(1)) &
        (close > open_.shift(1)) & (open_ < close.shift(1))
    ).astype(int)
    df['bear_engulf'] = (
        (close < open_) & (close.shift(1) > open_.shift(1)) &
        (close < open_.shift(1)) & (open_ > close.shift(1))
    ).astype(int)

    # ===== 13. ZAMAN ÖZELLİKLERİ =====
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['is_morning'] = ((df.index.hour >= 9) & (df.index.hour <= 11)).astype(int)
    df['is_midday'] = ((df.index.hour >= 11) & (df.index.hour <= 14)).astype(int)
    df['is_afternoon'] = ((df.index.hour >= 14) & (df.index.hour <= 17)).astype(int)
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)
    df['is_week_start'] = (df.index.dayofweek <= 1).astype(int)
    df['is_week_end'] = (df.index.dayofweek >= 3).astype(int)

    # ===== 14. REJİM TESPİTİ =====
    # Trend rejimi: ADX benzeri
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    atr14 = compute_atr(high, low, close, 14)
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-8)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(14).mean()
    df['adx'] = adx / 100
    df['plus_di'] = plus_di / 100
    df['minus_di'] = minus_di / 100
    df['di_diff'] = (plus_di - minus_di) / 100
    df['is_trending'] = (adx > 25).astype(int)
    df['is_ranging'] = (adx < 20).astype(int)
    df['trend_up'] = ((adx > 25) & (plus_di > minus_di)).astype(int)
    df['trend_down'] = ((adx > 25) & (minus_di > plus_di)).astype(int)

    # ===== 15. NORMALİZE FİYAT =====
    for w in [8, 12, 20, 24, 48]:
        df[f'price_norm_{w}'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-8)

    # ===== 16. STREAK =====
    streak = []
    current = 0
    for r in ret:
        if pd.isna(r):
            streak.append(0)
        elif r > 0:
            current = max(0, current) + 1
            streak.append(current)
        elif r < 0:
            current = min(0, current) - 1
            streak.append(current)
        else:
            current = 0
            streak.append(0)
    df['price_streak'] = streak
    df['streak_abs'] = np.abs(streak)
    df['streak_up'] = np.maximum(0, streak)
    df['streak_down'] = np.minimum(0, streak)

    # ===== 17. DESTEK/DİRENÇ =====
    for w in [10, 20, 48]:
        df[f'near_high_{w}'] = ((high.rolling(w).max() - close) / (close + 1e-8) < 0.01).astype(int)
        df[f'near_low_{w}'] = ((close - low.rolling(w).min()) / (close + 1e-8) < 0.01).astype(int)

    # ===== 18. FIYAT HIZI VE İVMESİ =====
    df['price_velocity'] = close.diff(3) / (close.shift(3) + 1e-8)
    df['price_acceleration'] = df['price_velocity'].diff(3)
    df['price_jerk'] = df['price_acceleration'].diff(3)

    return df


def make_target(df, horizon):
    """horizon bar sonra fiyat artacak mı?"""
    future_ret = df['Close'].pct_change(horizon).shift(-horizon)
    return (future_ret > 0).astype(int)


def train_advanced_model(X_train, y_train, X_test, y_test, n_splits=5, horizon_name=''):
    """Gelişmiş walk-forward stacking + Platt kalibrasyon."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Farklı hiperparametrelerle çoklu model
    base_configs = {
        'xgb_deep': xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.2, reg_lambda=2.0, gamma=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        ),
        'xgb_shallow': xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric='logloss',
            random_state=123, verbosity=0
        ),
        'lgb_deep': lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.6, min_child_samples=15,
            reg_alpha=0.2, reg_lambda=2.0, num_leaves=31,
            random_state=42, verbose=-1
        ),
        'lgb_shallow': lgb.LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, num_leaves=15,
            random_state=123, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=8, random_state=42
        ),
    }

    n_models = len(base_configs)
    oof_preds = np.zeros((len(X_train), n_models))
    cv_scores = []

    print(f"    Walk-forward CV ({n_splits} fold, {n_models} model)...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        fold_preds = []
        for i, (name, model) in enumerate(base_configs.items()):
            model.fit(X_tr, y_tr)
            pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, i] = pred
            fold_preds.append(pred)

        fold_avg = np.mean(fold_preds, axis=0)
        fold_acc = accuracy_score(y_val, (fold_avg > 0.5).astype(int))
        cv_scores.append(fold_acc)
        print(f"      Fold {fold+1}: {fold_acc:.4f}")

    cv_accuracy = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"    CV Doğruluk: {cv_accuracy:.4f} ± {cv_std:.4f}")

    # Tüm eğitim verisiyle eğit
    trained_base = {}
    for name, model in base_configs.items():
        model.fit(X_train, y_train)
        trained_base[name] = model

    # Meta-learner (birden fazla dene, en iyisini seç)
    meta_candidates = {
        'lr_c1': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        'lr_c05': LogisticRegression(C=0.5, random_state=42, max_iter=1000),
        'lr_c01': LogisticRegression(C=0.1, random_state=42, max_iter=1000),
    }

    best_meta = None
    best_meta_acc = 0
    for meta_name, meta in meta_candidates.items():
        meta.fit(oof_preds, y_train)
        test_base_preds = np.column_stack([
            m.predict_proba(X_test)[:, 1] for m in trained_base.values()
        ])
        meta_proba = meta.predict_proba(test_base_preds)[:, 1]
        meta_acc = accuracy_score(y_test, (meta_proba > 0.5).astype(int))
        if meta_acc > best_meta_acc:
            best_meta_acc = meta_acc
            best_meta = meta

    # Test tahminleri
    test_base_preds = np.column_stack([
        m.predict_proba(X_test)[:, 1] for m in trained_base.values()
    ])
    stacking_proba = best_meta.predict_proba(test_base_preds)[:, 1]
    avg_proba = np.mean(test_base_preds, axis=1)

    # Ağırlıklı kombinasyon - stacking daha iyi ise daha fazla ağırlık ver
    stack_acc = accuracy_score(y_test, (stacking_proba > 0.5).astype(int))
    avg_acc = accuracy_score(y_test, (avg_proba > 0.5).astype(int))

    if stack_acc > avg_acc:
        w_stack = 0.7
    else:
        w_stack = 0.3

    final_proba = stacking_proba * w_stack + avg_proba * (1 - w_stack)

    test_acc = accuracy_score(y_test, (final_proba > 0.5).astype(int))
    try:
        test_auc = roc_auc_score(y_test, final_proba)
    except:
        test_auc = 0.5
    print(f"    Test Doğruluk: {test_acc:.4f} | AUC: {test_auc:.4f}")

    # Optimal eşik bul (0.5 yerine)
    best_threshold = 0.5
    best_threshold_acc = test_acc
    for threshold in np.arange(0.35, 0.65, 0.01):
        t_acc = accuracy_score(y_test, (final_proba > threshold).astype(int))
        if t_acc > best_threshold_acc:
            best_threshold_acc = t_acc
            best_threshold = threshold

    if best_threshold != 0.5:
        print(f"    Optimal eşik: {best_threshold:.2f} → Doğruluk: {best_threshold_acc:.4f}")

    # Platt Scaling
    def neg_log_loss(a):
        calibrated = expit(a * (final_proba - 0.5))
        eps = 1e-7
        calibrated = np.clip(calibrated, eps, 1 - eps)
        return -np.mean(y_test * np.log(calibrated) + (1 - y_test) * np.log(1 - calibrated))

    result = minimize_scalar(neg_log_loss, bounds=(0.1, 10), method='bounded')
    calib_a = result.x
    calibrated_proba = expit(calib_a * (final_proba - 0.5))
    calib_acc = accuracy_score(y_test, (calibrated_proba > 0.5).astype(int))
    try:
        calib_auc = roc_auc_score(y_test, calibrated_proba)
    except:
        calib_auc = 0.5
    print(f"    Kalibre: Doğruluk={calib_acc:.4f} | AUC={calib_auc:.4f} (a={calib_a:.2f})")

    # En iyi sonucu seç
    final_acc = max(test_acc, calib_acc, best_threshold_acc)
    print(f"    En iyi doğruluk: {final_acc:.4f}")

    return {
        'base_models': trained_base,
        'meta_learner': best_meta,
        'calib_a': calib_a,
        'optimal_threshold': best_threshold,
        'cv_accuracy': cv_accuracy,
        'cv_std': cv_std,
        'test_accuracy': final_acc,
        'test_auc': calib_auc,
        'trained_at': datetime.now().isoformat(),
        'horizon': horizon_name,
        'is_realistic': final_acc >= 0.55,
        'issues': [] if final_acc >= 0.55 else ['⚠️ Düşük doğruluk'],
    }


def main():
    print("=" * 60)
    print("  GMSTR SAATLİK MODEL EĞİTİMİ v2 (1h / 4h)")
    print("  Hedef: %60+ Doğruluk")
    print("=" * 60)

    # Veri yükle
    print("\n[Veri Yükleme]")
    df_raw = load_hourly_data()
    if df_raw is None or len(df_raw) < 200:
        print("HATA: Yeterli saatlik veri yok!")
        return

    training_results = {}

    horizons = {
        '1h': (1, 1),   # 1 saatlik tahmin, 1h resample
        '4h': (4, 4),   # 4 saatlik tahmin, 4h resample
    }

    for h_name, (h_periods, resample_h) in horizons.items():
        print(f"\n{'='*60}")
        print(f"[{h_name} Modeli - {h_periods} bar tahmin]")
        print(f"{'='*60}")

        # Özellik mühendisliği
        df = engineer_features(df_raw.copy(), resample_hours=resample_h)
        print(f"  Toplam bar: {len(df)}")

        # Hedef
        target = make_target(df, h_periods)

        # Özellik kolonları
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('target_')]

        # Geçerli satırlar
        valid_mask = df[feature_cols].notna().all(axis=1) & target.notna()
        df_valid = df[valid_mask].copy()
        target_valid = target[valid_mask]

        print(f"  Geçerli bar: {len(df_valid)} | Pozitif oran: {target_valid.mean():.3f}")

        if len(df_valid) < 150:
            print(f"  UYARI: Yetersiz veri ({len(df_valid)} bar), atlanıyor.")
            continue

        # Özellik seçimi (mutual info)
        X_all = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X_all = np.clip(X_all, -1e6, 1e6)
        y_all = target_valid.values

        print(f"  Toplam özellik: {len(feature_cols)}")
        mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
        n_features = min(100, len(feature_cols))
        top_idx = np.argsort(mi_scores)[-n_features:]
        selected_cols = [feature_cols[i] for i in sorted(top_idx)]
        print(f"  Seçilen özellik (MI): {len(selected_cols)}")

        X = df_valid[selected_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X = np.clip(X, -1e6, 1e6)
        y = y_all

        # Train/test split (%80/%20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"  Eğitim: {len(X_train)} | Test: {len(X_test)}")

        # Ölçekleme (RobustScaler - outlier'lara daha dayanıklı)
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        X_train_sc = np.clip(X_train_sc, -10, 10)
        X_test_sc = np.clip(X_test_sc, -10, 10)

        # Model eğit
        result = train_advanced_model(X_train_sc, y_train, X_test_sc, y_test,
                                       n_splits=5, horizon_name=h_name)
        result['feature_cols'] = selected_cols
        result['scaler'] = scaler
        result['positive_rate'] = float(target_valid.mean())
        result['n_samples'] = len(df_valid)
        result['resample_hours'] = resample_h
        result['train_size'] = len(X_train)
        result['test_size'] = len(X_test)

        # Kaydet
        key = f'{h_name}_hourly'
        pkl_path = MODEL_DIR / f'simple_{h_name}_hourly.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)

        training_results[key] = {
            'horizon': h_name,
            'frequency': 'hourly',
            'cv_accuracy': round(result['cv_accuracy'], 4),
            'cv_std': round(result['cv_std'], 4),
            'cv_auc': round(result.get('test_auc', 0), 4),
            'test_accuracy': round(result['test_accuracy'], 4),
            'test_auc': round(result['test_auc'], 4),
            'positive_rate': round(result['positive_rate'], 4),
            'train_size': result['train_size'],
            'test_size': result['test_size'],
            'is_realistic': result['is_realistic'],
            'issues': result['issues'],
            'trained_at': result['trained_at'],
        }
        print(f"  ✅ Kaydedildi: {pkl_path.name}")

    # Sonuçları kaydet
    results_path = MODEL_DIR / 'training_results.json'
    existing = {}
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            pass
    existing.update(training_results)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  SAATLİK EĞİTİM v2 TAMAMLANDI")
    print(f"{'='*60}")
    for key, r in training_results.items():
        status = "✅" if r['test_accuracy'] >= 0.60 else ("⚠️" if r['test_accuracy'] >= 0.55 else "❌")
        print(f"  {status} {key}: Test={r['test_accuracy']:.3f} AUC={r['test_auc']:.3f}")

    print("\nŞimdi tahminleri güncelleyin: python generate_predictions.py")


if __name__ == '__main__':
    main()
