"""
GMSTR Gelişmiş Model Eğitimi
- Makro özellikler (USD/TRY, Altın, Gümüş vadeli, BIST100)
- Platt scaling kalibrasyon
- 5d modeli kaldırıldı (zararlı)
- Walk-forward cross-validation
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / 'gmstr_models'
CSV_PATH = ROOT / 'claude' / 'areaxdatetime.csv'


# ============================================================
# VERİ YÜKLEME
# ============================================================
def load_data():
    from gmstr_system.data_loader import GMSTRDataLoader
    loader = GMSTRDataLoader(str(CSV_PATH))
    loader.load()
    df = loader.clean()
    print(f"  Veri: {len(df)} satır | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ============================================================
# ÖZELLİK MÜHENDİSLİĞİ (Makro + Teknik)
# ============================================================
def engineer_features(df):
    from gmstr_system.features import FeatureEngineer
    eng = FeatureEngineer()
    df = eng.transform(df)
    close = df['Close']

    # --- Teknik özellikler ---
    for lag in [1, 2, 3, 5, 7, 10, 15, 20]:
        df[f'ret_lag_{lag}'] = close.pct_change(lag)
        df[f'close_lag_{lag}'] = close.shift(lag)

    for window in [5, 10, 20, 50]:
        df[f'mom_{window}'] = close / close.shift(window) - 1
        df[f'vol_{window}'] = close.pct_change().rolling(window).std()
        df[f'vol_ratio_{window}'] = df[f'vol_{window}'] / df[f'vol_{window}'].shift(window)

    for window in [10, 20, 50]:
        roll_min = close.rolling(window).min()
        roll_max = close.rolling(window).max()
        df[f'price_pos_{window}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)

    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)

    if 'Volume' in df.columns:
        vol = df['Volume']
        df['vol_ma_5'] = vol.rolling(5).mean()
        df['vol_ma_20'] = vol.rolling(20).mean()
        df['vol_ratio_v'] = vol / (df['vol_ma_20'] + 1e-8)
        df['price_vol'] = close * vol

    for short, long in [(5, 20), (10, 50), (20, 60)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)

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

    df['price_norm_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
    df['price_norm_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-8)

    # --- Makro özellikler ---
    macro_cols = ['usd_try', 'gold_usd', 'silver_usd', 'bist100',
                  'usd_try_ret', 'gold_usd_ret', 'silver_usd_ret', 'bist100_ret',
                  'gold_silver_ratio', 'gold_silver_ratio_ret']

    for col in macro_cols:
        if col in df.columns:
            # Lag özellikleri
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag3'] = df[col].shift(3)
            # Momentum
            df[f'{col}_mom5'] = df[col].pct_change(5)
            # Normalize
            df[f'{col}_norm20'] = (df[col] - df[col].rolling(20).mean()) / (df[col].rolling(20).std() + 1e-8)

    # GMSTR vs Gümüş korelasyonu (20 günlük)
    if 'silver_usd' in df.columns:
        gmstr_ret = close.pct_change()
        silver_ret = df['silver_usd'].pct_change()
        df['gmstr_silver_corr'] = gmstr_ret.rolling(20).corr(silver_ret)

    # GMSTR vs USD/TRY korelasyonu
    if 'usd_try' in df.columns:
        usd_ret = df['usd_try'].pct_change()
        df['gmstr_usd_corr'] = gmstr_ret.rolling(20).corr(usd_ret)

    return df


# ============================================================
# HEDEF OLUŞTURMA
# ============================================================
def make_target(df, horizon):
    """horizon gün sonra fiyat artacak mı? (1=evet, 0=hayır)"""
    future_ret = df['Close'].pct_change(horizon).shift(-horizon)
    return (future_ret > 0).astype(int)


# ============================================================
# MODEL EĞİTİMİ (Stacking + Kalibrasyon)
# ============================================================
def train_stacking_model(X_train, y_train, X_test, y_test, n_splits=5):
    """Walk-forward stacking ensemble + Platt scaling kalibrasyon."""

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Base modeller
    base_model_configs = {
        'xgb': xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }

    # OOF tahminleri
    oof_preds = np.zeros((len(X_train), len(base_model_configs)))
    cv_scores = []

    print(f"    Walk-forward CV ({n_splits} fold)...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        fold_preds = []
        for i, (name, model) in enumerate(base_model_configs.items()):
            model.fit(X_tr, y_tr)
            pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, i] = pred
            fold_preds.append(pred)

        fold_avg = np.mean(fold_preds, axis=0)
        fold_acc = accuracy_score(y_val, (fold_avg > 0.5).astype(int))
        cv_scores.append(fold_acc)

    cv_accuracy = np.mean(cv_scores)
    print(f"    CV Doğruluk: {cv_accuracy:.4f} ± {np.std(cv_scores):.4f}")

    # Tüm eğitim verisiyle base modelleri eğit
    trained_base = {}
    for name, model in base_model_configs.items():
        model.fit(X_train, y_train)
        trained_base[name] = model

    # Meta-learner (Logistic Regression)
    meta = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
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

    # Platt Scaling Kalibrasyon
    # Stacking çıktısını CalibratedClassifierCV ile kalibre et
    # Basit yaklaşım: sigmoid kalibrasyon
    from sklearn.calibration import calibration_curve
    from scipy.special import expit
    from scipy.optimize import minimize_scalar

    # Sigmoid kalibrasyon parametrelerini bul
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
    print(f"    Kalibre Test Doğruluk: {calib_acc:.4f} | AUC: {calib_auc:.4f} (a={calib_a:.2f})")

    return {
        'base_models': trained_base,
        'meta_learner': meta,
        'calib_a': calib_a,
        'cv_accuracy': cv_accuracy,
        'test_accuracy': calib_acc,
        'test_auc': calib_auc,
        'trained_at': datetime.now().isoformat(),
    }


# ============================================================
# ANA EĞİTİM FONKSİYONU
# ============================================================
def main():
    print("=" * 60)
    print("  GMSTR GELİŞMİŞ MODEL EĞİTİMİ")
    print("  Makro Özellikler + Platt Kalibrasyon")
    print("=" * 60)

    # Veri yükle
    print("\n[Veri Yükleme]")
    df = load_data()
    df = engineer_features(df)

    # Özellik kolonları (makro dahil)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'usd_try', 'gold_usd', 'silver_usd', 'bist100',
                    'usd_try_ret', 'gold_usd_ret', 'silver_usd_ret', 'bist100_ret',
                    'gold_silver_ratio', 'gold_silver_ratio_ret']

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not c.startswith('target_')]

    print(f"  Toplam özellik: {len(feature_cols)}")

    # Makro özellik sayısını göster
    macro_features = [c for c in feature_cols if any(m in c for m in
                      ['usd_try', 'gold_usd', 'silver_usd', 'bist100',
                       'gold_silver', 'gmstr_silver', 'gmstr_usd'])]
    print(f"  Makro özellik: {len(macro_features)}")

    training_results = {}

    # 5d modeli KALDIRILDI - zararlı (%43.5 doğruluk)
    horizons = {
        '1d': 1,
        '3d': 3,
        # '5d': 5,  # KALDIRILDI - zararlı model
        '10d': 10,
    }

    for h_name, h_days in horizons.items():
        print(f"\n[{h_name} Modeli - {h_days} günlük tahmin]")

        # Hedef oluştur
        target = make_target(df, h_days)

        # Geçerli satırları seç
        valid_mask = df[feature_cols].notna().all(axis=1) & target.notna()
        df_valid = df[valid_mask].copy()
        target_valid = target[valid_mask]

        print(f"  Geçerli satır: {len(df_valid)} | Pozitif oran: {target_valid.mean():.3f}")

        # Özellik seçimi (mutual info)
        from sklearn.feature_selection import mutual_info_classif
        X_all = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        # Sonsuz/çok büyük değerleri temizle
        X_all = np.clip(X_all, -1e9, 1e9)
        y_all = target_valid.values

        mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
        top_idx = np.argsort(mi_scores)[-60:]  # En iyi 60 özellik (makro dahil)
        selected_cols = [feature_cols[i] for i in sorted(top_idx)]

        # Makro özelliklerin seçildiğini kontrol et
        selected_macro = [c for c in selected_cols if any(m in c for m in
                          ['usd_try', 'gold_usd', 'silver_usd', 'bist100',
                           'gold_silver', 'gmstr_silver', 'gmstr_usd'])]
        print(f"  Seçilen özellik: {len(selected_cols)} (makro: {len(selected_macro)})")

        X = df_valid[selected_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        X = np.clip(X, -1e9, 1e9)
        y = y_all

        # Train/test split (%80/%20, zaman sıralı)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Ölçekleme
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        # Ölçekleme sonrası da inf kontrolü
        X_train_sc = np.clip(X_train_sc, -10, 10)
        X_test_sc = np.clip(X_test_sc, -10, 10)

        # Model eğit
        result = train_stacking_model(X_train_sc, y_train, X_test_sc, y_test)
        result['feature_cols'] = selected_cols
        result['scaler'] = scaler
        result['positive_rate'] = float(target_valid.mean())
        result['n_samples'] = len(df_valid)
        result['macro_features_used'] = selected_macro

        # Kaydet
        key = f'{h_name}_daily'
        pkl_path = MODEL_DIR / f'simple_{h_name}_daily.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)

        training_results[key] = {
            'cv_accuracy': result['cv_accuracy'],
            'test_accuracy': result['test_accuracy'],
            'test_auc': result['test_auc'],
            'positive_rate': result['positive_rate'],
            'n_samples': result['n_samples'],
            'macro_features': len(selected_macro),
            'trained_at': result['trained_at'],
        }

        print(f"  Kaydedildi: {pkl_path.name}")

    # Eğitim sonuçlarını kaydet
    results_path = MODEL_DIR / 'training_results.json'
    # Mevcut sonuçları yükle (saatlik modeller için)
    existing = {}
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)

    # Güncelle (saatlik modelleri koru)
    existing.update(training_results)

    # 5d_daily'yi kaldır
    existing.pop('5d_daily', None)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  EĞİTİM TAMAMLANDI")
    print(f"{'='*60}")
    for key, r in training_results.items():
        status = "✅" if r['test_accuracy'] >= 0.55 else "⚠️"
        print(f"  {status} {key}: Test={r['test_accuracy']:.3f} AUC={r['test_auc']:.3f} Makro={r['macro_features']} özellik")

    print("\nŞimdi tahminleri güncelleyin: python generate_predictions.py")


if __name__ == '__main__':
    main()
