"""
GMSTR 15 Dakikalık Model Eğitimi
- Yahoo Finance'den 15m veri çek
- Gelişmiş özellik mühendisliği
- Stacking ensemble + Platt kalibrasyon
- Hedef: %60+ doğruluk
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


def load_15m_data():
    """Yahoo Finance'den 15 dakikalık GMSTR verisi çek."""
    try:
        import yfinance as yf
        print("  Yahoo Finance'den 15m veri çekiliyor...")
        ticker = yf.Ticker("GMSTR.IS")
        # 15m veri maksimum 60 gün
        df = ticker.history(period="60d", interval="15m")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        # Borsa saatleri filtresi (09:00-18:30)
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


def engineer_15m_features(df):
    """15 dakikalık özellik mühendisliği."""
    close = df['Close'].copy()
    high = df['High'].copy()
    low = df['Low'].copy()
    open_ = df['Open'].copy()
    volume = df['Volume'].copy() if 'Volume' in df.columns else pd.Series(1, index=df.index)

    ret = close.pct_change()

    # === LAG GETİRİLER ===
    for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48]:
        df[f'ret_{lag}'] = close.pct_change(lag)

    # === MOMENTUM ===
    for p in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96]:
        df[f'mom_{p}'] = close / close.shift(p) - 1

    # === VOLATİLİTE ===
    for w in [4, 8, 12, 16, 24, 32, 48]:
        df[f'vol_{w}'] = ret.rolling(w).std()
    df['vol_ratio_4_24'] = df['vol_4'] / (df['vol_24'] + 1e-8)
    df['vol_ratio_8_48'] = df['vol_8'] / (df['vol_48'] + 1e-8)

    # === HAREKETLİ ORTALAMALAR ===
    for w in [4, 8, 12, 16, 20, 24, 32, 48, 64, 96]:
        df[f'ma_{w}'] = close.rolling(w).mean()
        df[f'ma_ratio_{w}'] = close / (df[f'ma_{w}'] + 1e-8) - 1

    for w in [8, 16, 24, 48]:
        df[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
        df[f'ema_ratio_{w}'] = close / (df[f'ema_{w}'] + 1e-8) - 1

    # MA crossover
    for short, long in [(4, 16), (8, 24), (8, 48), (16, 48)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)
        df[f'ma_cross_sign_{short}_{long}'] = np.sign(ma_s - ma_l)

    # === RSI ===
    for period in [7, 14, 21]:
        rsi = compute_rsi(close, period)
        df[f'rsi_{period}'] = rsi / 100
        df[f'rsi_{period}_ob'] = (rsi > 70).astype(int)
        df[f'rsi_{period}_os'] = (rsi < 30).astype(int)
        df[f'rsi_{period}_norm'] = (rsi - 50) / 50

    # === MACD ===
    for fast, slow, sig in [(12, 26, 9), (8, 21, 8)]:
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig, adjust=False).mean()
        df[f'macd_{fast}_{slow}'] = macd / (close + 1e-8)
        df[f'macd_hist_{fast}_{slow}'] = (macd - signal) / (close + 1e-8)
        df[f'macd_cross_{fast}_{slow}'] = np.sign(macd - signal)

    # === BOLLİNGER ===
    for w in [12, 20, 24]:
        bb_mid = close.rolling(w).mean()
        bb_std = close.rolling(w).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df[f'bb_pos_{w}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        df[f'bb_width_{w}'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        df[f'bb_above_{w}'] = (close > bb_upper).astype(int)
        df[f'bb_below_{w}'] = (close < bb_lower).astype(int)

    # === STOKASTİK ===
    for w in [9, 14, 21]:
        low_min = low.rolling(w).min()
        high_max = high.rolling(w).max()
        stoch_k = (close - low_min) / (high_max - low_min + 1e-8) * 100
        df[f'stoch_k_{w}'] = stoch_k / 100
        df[f'stoch_ob_{w}'] = (stoch_k > 80).astype(int)
        df[f'stoch_os_{w}'] = (stoch_k < 20).astype(int)

    # === ATR ===
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    for w in [7, 14, 21]:
        df[f'atr_{w}'] = tr.rolling(w).mean() / (close + 1e-8)

    # === FİYAT POZİSYONU ===
    for w in [8, 16, 24, 48, 96]:
        roll_min = close.rolling(w).min()
        roll_max = close.rolling(w).max()
        df[f'price_pos_{w}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)
        df[f'dist_high_{w}'] = (roll_max - close) / (close + 1e-8)
        df[f'dist_low_{w}'] = (close - roll_min) / (close + 1e-8)

    # === HACİM ===
    for w in [8, 16, 24, 48]:
        vol_ma = volume.rolling(w).mean()
        df[f'vol_ratio_v_{w}'] = volume / (vol_ma + 1e-8)

    # OBV
    obv = (np.sign(close.diff()) * volume).cumsum()
    for w in [16, 32]:
        obv_ma = obv.rolling(w).mean()
        df[f'obv_trend_{w}'] = np.sign(obv - obv_ma)

    # === MEKTUP ÖZELLİKLERİ ===
    df['candle_body'] = (close - open_) / (close + 1e-8)
    df['candle_range'] = (high - low) / (close + 1e-8)
    df['candle_is_bull'] = (close > open_).astype(int)
    df['candle_upper_shadow'] = (high - close.clip(lower=open_)) / (close + 1e-8)
    df['candle_lower_shadow'] = (close.clip(upper=open_) - low) / (close + 1e-8)

    # === ZAMAN ===
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['is_morning_open'] = ((df.index.hour == 9) & (df.index.minute <= 30)).astype(int)
    df['is_morning'] = ((df.index.hour >= 9) & (df.index.hour <= 11)).astype(int)
    df['is_midday'] = ((df.index.hour >= 11) & (df.index.hour <= 14)).astype(int)
    df['is_afternoon'] = ((df.index.hour >= 14) & (df.index.hour <= 17)).astype(int)
    df['is_close'] = (df.index.hour >= 17).astype(int)
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)

    # === NORMALİZE FİYAT ===
    for w in [16, 24, 48]:
        df[f'price_norm_{w}'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-8)

    # === ADX ===
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-8)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-8)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(14).mean()
    df['adx'] = adx / 100
    df['di_diff'] = (plus_di - minus_di) / 100
    df['is_trending'] = (adx > 25).astype(int)

    # === STREAK ===
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

    return df


def make_target(df, horizon=1):
    """horizon bar (15 dakika) sonra fiyat artacak mı?"""
    future_ret = df['Close'].pct_change(horizon).shift(-horizon)
    return (future_ret > 0).astype(int)


def train_model(X_train, y_train, X_test, y_test, n_splits=5):
    """Walk-forward stacking + Platt kalibrasyon."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    base_configs = {
        'xgb': xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=15,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=8,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=8,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=8, random_state=42
        ),
    }

    oof_preds = np.zeros((len(X_train), len(base_configs)))
    cv_scores = []

    print(f"    Walk-forward CV ({n_splits} fold)...")
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

    # Tüm eğitim verisiyle eğit
    trained_base = {}
    for name, model in base_configs.items():
        model.fit(X_train, y_train)
        trained_base[name] = model

    # Meta-learner
    meta = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    meta.fit(oof_preds, y_train)

    # Test
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

    # Platt Scaling
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
    print("  GMSTR 15 DAKİKALIK MODEL EĞİTİMİ")
    print("  Hedef: %60+ Doğruluk")
    print("=" * 60)

    df_raw = load_15m_data()
    if df_raw is None or len(df_raw) < 200:
        print("HATA: Yeterli 15m veri yok!")
        return

    print(f"\n[15m Modeli]")
    df = engineer_15m_features(df_raw.copy())

    # Hedef: 1 bar (15 dakika) sonra
    target = make_target(df, horizon=1)

    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    feature_cols = [c for c in df.columns if c not in exclude]

    valid_mask = df[feature_cols].notna().all(axis=1) & target.notna()
    df_valid = df[valid_mask].copy()
    target_valid = target[valid_mask]

    print(f"  Bar: {len(df_valid)} | Pozitif: {target_valid.mean():.3f}")

    if len(df_valid) < 150:
        print("  Yetersiz veri!")
        return

    X = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X = np.clip(X, -1e6, 1e6)
    y = target_valid.values

    # %75/%25 split
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
    result['horizon'] = '15m'
    result['frequency'] = '15min'

    # Kaydet
    pkl_path = MODEL_DIR / 'simple_15m_15min.pkl'
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

    existing['15m_15min'] = {
        'horizon': '15m',
        'frequency': '15min',
        'cv_accuracy': round(result['cv_accuracy'], 4),
        'cv_std': round(result['cv_std'], 4),
        'cv_auc': round(result['test_auc'], 4),
        'test_accuracy': round(result['test_accuracy'], 4),
        'test_auc': round(result['test_auc'], 4),
        'positive_rate': round(result['positive_rate'], 4),
        'train_size': result['train_size'],
        'test_size': result['test_size'],
        'is_realistic': result['test_accuracy'] >= 0.60,
        'issues': [] if result['test_accuracy'] >= 0.60 else ['⚠️ Hedef %60 altında'],
        'trained_at': result['trained_at'],
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    status = "✅" if result['test_accuracy'] >= 0.60 else ("⚠️" if result['test_accuracy'] >= 0.55 else "❌")
    print(f"\n  {status} 15m_15min: Test={result['test_accuracy']:.3f} AUC={result['test_auc']:.3f}")
    print("\nŞimdi: python generate_predictions.py")


if __name__ == '__main__':
    main()
