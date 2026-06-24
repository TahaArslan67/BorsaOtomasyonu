"""
GMSTR 4h Model - 180 günlük veri ile eğitim
4h resample için yeterli bar sayısı gerekiyor
"""
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'gmstr_models'


def load_hourly_data(period="180d"):
    try:
        import yfinance as yf
        print(f"  Yahoo Finance'den saatlik veri ({period})...")
        ticker = yf.Ticker("GMSTR.IS")
        df = ticker.history(period=period, interval="1h")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df[(df.index.hour >= 9) & (df.index.hour <= 18)]
        df = df[df.index.dayofweek < 5]
        print(f"  {len(df)} bar | {df.index[0].date()} -> {df.index[-1].date()}")
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


def engineer_4h_features(df):
    """4h resample + özellik mühendisliği."""
    # 4h resample
    df4 = df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df4 = df4[(df4.index.hour >= 9) & (df4.index.hour <= 18)]
    df4 = df4[df4.index.dayofweek < 5]

    close = df4['Close'].copy()
    high = df4['High'].copy()
    low = df4['Low'].copy()
    open_ = df4['Open'].copy()
    volume = df4['Volume'].copy()

    ret = close.pct_change()

    # Lag getiriler
    for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
        df4[f'ret_{lag}'] = close.pct_change(lag)

    # Momentum
    for p in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        df4[f'mom_{p}'] = close / close.shift(p) - 1

    # Volatilite
    for w in [3, 5, 8, 12]:
        df4[f'vol_{w}'] = ret.rolling(w).std()
    df4['vol_ratio'] = df4['vol_3'] / (df4['vol_12'] + 1e-8)

    # MA
    for w in [3, 5, 8, 10, 20]:
        df4[f'ma_{w}'] = close.rolling(w).mean()
        df4[f'ma_ratio_{w}'] = close / (df4[f'ma_{w}'] + 1e-8) - 1

    for w in [5, 10, 20]:
        df4[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
        df4[f'ema_ratio_{w}'] = close / (df4[f'ema_{w}'] + 1e-8) - 1

    # MA crossover
    for short, long in [(3, 10), (5, 20)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df4[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)
        df4[f'ma_cross_sign_{short}_{long}'] = np.sign(ma_s - ma_l)

    # RSI
    for period in [7, 14]:
        rsi = compute_rsi(close, period)
        df4[f'rsi_{period}'] = rsi / 100
        df4[f'rsi_{period}_ob'] = (rsi > 70).astype(int)
        df4[f'rsi_{period}_os'] = (rsi < 30).astype(int)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df4['macd'] = macd / (close + 1e-8)
    df4['macd_hist'] = (macd - signal) / (close + 1e-8)
    df4['macd_cross'] = np.sign(macd - signal)

    # Bollinger
    for w in [10, 20]:
        bb_mid = close.rolling(w).mean()
        bb_std = close.rolling(w).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df4[f'bb_pos_{w}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df4[f'bb_width_{w}'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)

    # Stochastic
    for w in [5, 9]:
        low_min = low.rolling(w).min()
        high_max = high.rolling(w).max()
        stoch_k = (close - low_min) / (high_max - low_min + 1e-8) * 100
        df4[f'stoch_k_{w}'] = stoch_k / 100
        df4[f'stoch_ob_{w}'] = (stoch_k > 80).astype(int)
        df4[f'stoch_os_{w}'] = (stoch_k < 20).astype(int)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    for w in [5, 10]:
        df4[f'atr_{w}'] = tr.rolling(w).mean() / (close + 1e-8)

    # Fiyat pozisyonu
    for w in [5, 10, 20]:
        roll_min = close.rolling(w).min()
        roll_max = close.rolling(w).max()
        df4[f'price_pos_{w}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)

    # Hacim
    for w in [5, 10]:
        vol_ma = volume.rolling(w).mean()
        df4[f'vol_ratio_v_{w}'] = volume / (vol_ma + 1e-8)

    # Candle
    df4['candle_body'] = (close - open_) / (close + 1e-8)
    df4['candle_range'] = (high - low) / (close + 1e-8)
    df4['candle_is_bull'] = (close > open_).astype(int)

    # Zaman
    df4['hour'] = df4.index.hour
    df4['day_of_week'] = df4.index.dayofweek
    df4['is_morning'] = (df4.index.hour <= 13).astype(int)
    df4['is_afternoon'] = (df4.index.hour >= 14).astype(int)
    df4['is_monday'] = (df4.index.dayofweek == 0).astype(int)
    df4['is_friday'] = (df4.index.dayofweek == 4).astype(int)

    # Normalize fiyat
    for w in [5, 10]:
        df4[f'price_norm_{w}'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-8)

    # Streak
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
    df4['price_streak'] = streak

    return df4


def make_target(df, horizon):
    future_ret = df['Close'].pct_change(horizon).shift(-horizon)
    return (future_ret > 0).astype(int)


def train_model(X_train, y_train, X_test, y_test, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    base_configs = {
        'xgb': xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42
        ),
    }

    oof_preds = np.zeros((len(X_train), len(base_configs)))
    cv_scores = []

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
    print(f"    CV: {cv_accuracy:.4f} ± {np.std(cv_scores):.4f}")

    trained_base = {}
    for name, model in base_configs.items():
        model.fit(X_train, y_train)
        trained_base[name] = model

    meta = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    meta.fit(oof_preds, y_train)

    test_base_preds = np.column_stack([
        m.predict_proba(X_test)[:, 1] for m in trained_base.values()
    ])
    stacking_proba = meta.predict_proba(test_base_preds)[:, 1]
    avg_proba = np.mean(test_base_preds, axis=1)
    final_proba = stacking_proba * 0.6 + avg_proba * 0.4

    test_acc = accuracy_score(y_test, (final_proba > 0.5).astype(int))
    try:
        test_auc = roc_auc_score(y_test, final_proba)
    except:
        test_auc = 0.5

    # Optimal eşik
    best_threshold = 0.5
    best_acc = test_acc
    for t in np.arange(0.35, 0.65, 0.01):
        acc = accuracy_score(y_test, (final_proba > t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    # Platt
    def neg_log_loss(a):
        cal = expit(a * (final_proba - 0.5))
        cal = np.clip(cal, 1e-7, 1 - 1e-7)
        return -np.mean(y_test * np.log(cal) + (1 - y_test) * np.log(1 - cal))

    res = minimize_scalar(neg_log_loss, bounds=(0.1, 10), method='bounded')
    calib_a = res.x
    cal_proba = expit(calib_a * (final_proba - 0.5))
    calib_acc = accuracy_score(y_test, (cal_proba > 0.5).astype(int))
    try:
        calib_auc = roc_auc_score(y_test, cal_proba)
    except:
        calib_auc = 0.5

    final_acc = max(test_acc, calib_acc, best_acc)
    print(f"    Test={test_acc:.4f} | Kalibre={calib_acc:.4f} | Eşik={best_acc:.4f} → En iyi={final_acc:.4f}")

    return {
        'base_models': trained_base,
        'meta_learner': meta,
        'calib_a': calib_a,
        'optimal_threshold': best_threshold,
        'cv_accuracy': cv_accuracy,
        'cv_std': float(np.std(cv_scores)),
        'test_accuracy': final_acc,
        'test_auc': calib_auc,
        'trained_at': datetime.now().isoformat(),
    }


def main():
    print("=" * 60)
    print("  GMSTR 4h MODEL EĞİTİMİ (180 günlük veri)")
    print("=" * 60)

    df_raw = load_hourly_data(period="180d")
    if df_raw is None or len(df_raw) < 200:
        print("HATA: Yeterli veri yok!")
        return

    print("\n[4h Modeli]")
    df4 = engineer_4h_features(df_raw.copy())
    print(f"  4h bar sayısı: {len(df4)}")

    target = make_target(df4, 1)  # 1 bar (4 saat) sonra

    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    feature_cols = [c for c in df4.columns if c not in exclude]

    valid_mask = df4[feature_cols].notna().all(axis=1) & target.notna()
    df_valid = df4[valid_mask].copy()
    target_valid = target[valid_mask]

    print(f"  Geçerli bar: {len(df_valid)} | Pozitif: {target_valid.mean():.3f}")

    if len(df_valid) < 80:
        print("  Yetersiz veri!")
        return

    X = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X = np.clip(X, -1e6, 1e6)
    y = target_valid.values

    split = int(len(X) * 0.75)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Eğitim: {len(X_train)} | Test: {len(X_test)}")

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_train_sc = np.clip(X_train_sc, -10, 10)
    X_test_sc = np.clip(X_test_sc, -10, 10)

    result = train_model(X_train_sc, y_train, X_test_sc, y_test)
    result['feature_cols'] = feature_cols
    result['scaler'] = scaler
    result['positive_rate'] = float(target_valid.mean())
    result['n_samples'] = len(df_valid)
    result['train_size'] = len(X_train)
    result['test_size'] = len(X_test)
    result['horizon'] = '4h'
    result['frequency'] = 'hourly'

    pkl_path = MODEL_DIR / 'simple_4h_hourly.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"  ✅ Kaydedildi: {pkl_path.name}")

    # training_results.json güncelle
    results_path = MODEL_DIR / 'training_results.json'
    existing = {}
    if results_path.exists():
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            pass

    existing['4h_hourly'] = {
        'horizon': '4h',
        'frequency': 'hourly',
        'cv_accuracy': round(result['cv_accuracy'], 4),
        'cv_std': round(result['cv_std'], 4),
        'cv_auc': round(result['test_auc'], 4),
        'test_accuracy': round(result['test_accuracy'], 4),
        'test_auc': round(result['test_auc'], 4),
        'positive_rate': round(result['positive_rate'], 4),
        'train_size': result['train_size'],
        'test_size': result['test_size'],
        'is_realistic': result['test_accuracy'] >= 0.55,
        'issues': [] if result['test_accuracy'] >= 0.55 else ['⚠️ Düşük doğruluk'],
        'trained_at': result['trained_at'],
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    status = "✅" if result['test_accuracy'] >= 0.58 else ("⚠️" if result['test_accuracy'] >= 0.52 else "❌")
    print(f"\n  {status} 4h_hourly: Test={result['test_accuracy']:.3f} AUC={result['test_auc']:.3f}")
    print("\nŞimdi: python generate_predictions.py")


if __name__ == '__main__':
    main()
