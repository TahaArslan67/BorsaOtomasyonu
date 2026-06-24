"""
GMSTR Saatlik Model Eğitimi (1h / 4h)
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
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
        # Son 730 günlük saatlik veri (maksimum)
        df = ticker.history(period="730d", interval="1h")
        if len(df) < 100:
            print(f"  Yetersiz veri: {len(df)} satır, 60d deneniyor...")
            df = ticker.history(period="60d", interval="1h")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        print(f"  {len(df)} saatlik bar yüklendi | {df.index[0].date()} -> {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  Hata: {e}")
        return None


def engineer_hourly_features(df, resample_hours=1):
    """Gelişmiş saatlik özellik mühendisliği."""
    if resample_hours > 1:
        df = df.resample(f'{resample_hours}h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()

    close = df['Close'].copy()
    high = df['High'].copy()
    low = df['Low'].copy()
    volume = df['Volume'].copy() if 'Volume' in df.columns else pd.Series(0, index=df.index)

    # --- Lag özellikleri ---
    for lag in [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]:
        df[f'ret_lag_{lag}'] = close.pct_change(lag)
        df[f'close_lag_{lag}'] = close.shift(lag)

    # --- Momentum ---
    for p in [2, 3, 4, 6, 8, 12, 24, 48, 72]:
        df[f'mom_{p}'] = close / close.shift(p) - 1

    # --- Volatilite ---
    for w in [3, 6, 12, 24, 48]:
        df[f'vol_{w}'] = close.pct_change().rolling(w).std()
        df[f'vol_ratio_{w}'] = df[f'vol_{w}'] / df[f'vol_{w}'].shift(w)

    # --- Hareketli ortalamalar ---
    for w in [3, 5, 8, 13, 21, 34, 55]:
        df[f'ma_{w}'] = close.rolling(w).mean()
        df[f'ma_ratio_{w}'] = close / (df[f'ma_{w}'] + 1e-8) - 1

    # --- EMA ---
    for w in [5, 10, 20, 50]:
        df[f'ema_{w}'] = close.ewm(span=w).mean()
        df[f'ema_ratio_{w}'] = close / (df[f'ema_{w}'] + 1e-8) - 1

    # --- MA crossover ---
    for short, long in [(3, 12), (5, 20), (8, 21), (12, 48)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)
        df[f'ma_cross_sign_{short}_{long}'] = np.sign(ma_s - ma_l)

    # --- RSI ---
    for period in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}_norm'] = (df[f'rsi_{period}'] - 50) / 50

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd'] = macd / (close + 1e-8)
    df['macd_signal'] = signal / (close + 1e-8)
    df['macd_hist'] = (macd - signal) / (close + 1e-8)
    df['macd_cross'] = np.sign(macd - signal)

    # --- Bollinger Bands ---
    for w in [12, 20]:
        bb_mid = close.rolling(w).mean()
        bb_std = close.rolling(w).std()
        df[f'bb_upper_{w}'] = (bb_mid + 2 * bb_std - close) / (close + 1e-8)
        df[f'bb_lower_{w}'] = (close - (bb_mid - 2 * bb_std)) / (close + 1e-8)
        df[f'bb_pos_{w}'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-8)
        df[f'bb_width_{w}'] = 4 * bb_std / (bb_mid + 1e-8)

    # --- Stochastic ---
    for w in [9, 14]:
        low_min = low.rolling(w).min()
        high_max = high.rolling(w).max()
        df[f'stoch_k_{w}'] = (close - low_min) / (high_max - low_min + 1e-8)
        df[f'stoch_d_{w}'] = df[f'stoch_k_{w}'].rolling(3).mean()

    # --- ATR (Average True Range) ---
    for w in [7, 14]:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df[f'atr_{w}'] = tr.rolling(w).mean() / (close + 1e-8)

    # --- Fiyat pozisyonu ---
    for w in [6, 12, 24, 48, 96]:
        roll_min = close.rolling(w).min()
        roll_max = close.rolling(w).max()
        df[f'price_pos_{w}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)

    # --- Hacim özellikleri ---
    for w in [6, 12, 24]:
        vol_ma = volume.rolling(w).mean()
        df[f'vol_ratio_v_{w}'] = volume / (vol_ma + 1e-8)
    df['price_vol'] = close.pct_change() * volume

    # --- OBV (On Balance Volume) ---
    obv = (np.sign(close.diff()) * volume).cumsum()
    df['obv_norm'] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-8)

    # --- Zaman özellikleri ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_morning'] = ((df.index.hour >= 9) & (df.index.hour <= 11)).astype(int)
    df['is_midday'] = ((df.index.hour >= 11) & (df.index.hour <= 14)).astype(int)
    df['is_afternoon'] = ((df.index.hour >= 14) & (df.index.hour <= 17)).astype(int)
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)

    # --- Normalize fiyat ---
    for w in [12, 24, 48]:
        df[f'price_norm_{w}'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-8)

    # --- Streak ---
    daily_ret = close.pct_change()
    streak = []
    current = 0
    for r in daily_ret:
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

    # --- Candle özellikleri ---
    df['candle_body'] = (close - df['Open']) / (close + 1e-8)
    df['candle_upper_shadow'] = (high - close.clip(lower=df['Open'])) / (close + 1e-8)
    df['candle_lower_shadow'] = (close.clip(upper=df['Open']) - low) / (close + 1e-8)
    df['candle_range'] = (high - low) / (close + 1e-8)

    return df


def make_target(df, horizon):
    """horizon bar sonra fiyat artacak mı?"""
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
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=10, random_state=42
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
    print(f"    CV Doğruluk: {cv_accuracy:.4f} ± {np.std(cv_scores):.4f}")

    # Tüm eğitim verisiyle eğit
    trained_base = {}
    for name, model in base_configs.items():
        model.fit(X_train, y_train)
        trained_base[name] = model

    # Meta-learner
    meta = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
    meta.fit(oof_preds, y_train)

    # Test tahminleri
    test_base_preds = np.column_stack([
        m.predict_proba(X_test)[:, 1] for m in trained_base.values()
    ])
    stacking_proba = meta.predict_proba(test_base_preds)[:, 1]
    avg_proba = np.mean(test_base_preds, axis=1)
    final_proba = stacking_proba * 0.6 + avg_proba * 0.4

    test_acc = accuracy_score(y_test, (final_proba > 0.5).astype(int))
    test_auc = roc_auc_score(y_test, final_proba)
    print(f"    Test Doğruluk: {test_acc:.4f} | AUC: {test_auc:.4f}")

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
    calib_auc = roc_auc_score(y_test, calibrated_proba)
    print(f"    Kalibre: Doğruluk={calib_acc:.4f} | AUC={calib_auc:.4f} (a={calib_a:.2f})")

    return {
        'base_models': trained_base,
        'meta_learner': meta,
        'calib_a': calib_a,
        'cv_accuracy': cv_accuracy,
        'test_accuracy': calib_acc,
        'test_auc': calib_auc,
        'trained_at': datetime.now().isoformat(),
    }


def main():
    print("=" * 60)
    print("  GMSTR SAATLİK MODEL EĞİTİMİ (1h / 4h)")
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
        print(f"\n[{h_name} Modeli - {h_periods} bar tahmin]")

        # Özellik mühendisliği
        df = engineer_hourly_features(df_raw.copy(), resample_hours=resample_h)
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

        if len(df_valid) < 200:
            print(f"  UYARI: Yetersiz veri ({len(df_valid)} bar), atlanıyor.")
            continue

        # Özellik seçimi (mutual info)
        X_all = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X_all = np.clip(X_all, -1e6, 1e6)
        y_all = target_valid.values

        mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
        n_features = min(80, len(feature_cols))
        top_idx = np.argsort(mi_scores)[-n_features:]
        selected_cols = [feature_cols[i] for i in sorted(top_idx)]
        print(f"  Seçilen özellik: {len(selected_cols)}")

        X = df_valid[selected_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X = np.clip(X, -1e6, 1e6)
        y = y_all

        # Train/test split (%80/%20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Ölçekleme
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        X_train_sc = np.clip(X_train_sc, -10, 10)
        X_test_sc = np.clip(X_test_sc, -10, 10)

        # Model eğit
        result = train_model(X_train_sc, y_train, X_test_sc, y_test)
        result['feature_cols'] = selected_cols
        result['scaler'] = scaler
        result['positive_rate'] = float(target_valid.mean())
        result['n_samples'] = len(df_valid)
        result['resample_hours'] = resample_h

        # Kaydet
        key = f'{h_name}_hourly'
        pkl_path = MODEL_DIR / f'simple_{h_name}_hourly.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)

        training_results[key] = {
            'cv_accuracy': result['cv_accuracy'],
            'test_accuracy': result['test_accuracy'],
            'test_auc': result['test_auc'],
            'positive_rate': result['positive_rate'],
            'n_samples': result['n_samples'],
            'trained_at': result['trained_at'],
        }
        print(f"  Kaydedildi: {pkl_path.name}")

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
    print("  SAATLİK EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    for key, r in training_results.items():
        status = "✅" if r['test_accuracy'] >= 0.55 else "⚠️"
        print(f"  {status} {key}: Test={r['test_accuracy']:.3f} AUC={r['test_auc']:.3f}")

    print("\nŞimdi tahminleri güncelleyin: python generate_predictions.py")


if __name__ == '__main__':
    main()
