#!/usr/bin/env python3
"""
GMSTR Saatlik Model Eğitimi (1h ve 4h)
=======================================
Yahoo Finance'den saatlik veri çeker ve 1h/4h modelleri eğitir.
Kullanım: python retrain_hourly_models.py
"""
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / 'gmstr_models'
MODEL_DIR.mkdir(exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# VERİ YÜKLEME
# ============================================================
def load_hourly_data():
    """Yahoo Finance'den saatlik GMSTR verisi çek."""
    print("[Veri] Yahoo Finance'den saatlik veri çekiliyor...")
    try:
        import yfinance as yf
        ticker = yf.Ticker("GMSTR.IS")
        df = ticker.history(period="60d", interval="1h")
        if len(df) < 100:
            print(f"  ⚠ Yetersiz veri: {len(df)} satır (min 100 gerekli)")
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        print(f"  ✅ {len(df)} saatlik bar yüklendi | {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  ❌ Hata: {e}")
        return None


# ============================================================
# ÖZELLİK MÜHENDİSLİĞİ
# ============================================================
def engineer_hourly_features(df):
    """Saatlik veri için kapsamlı özellik mühendisliği."""
    close = df['Close'].copy()
    high = df['High'].copy() if 'High' in df.columns else close
    low = df['Low'].copy() if 'Low' in df.columns else close

    # --- Lag özellikleri ---
    for lag in [1, 2, 3, 4, 6, 8, 12, 24]:
        df[f'ret_lag_{lag}'] = close.pct_change(lag)
        df[f'close_lag_{lag}'] = close.shift(lag)

    # --- Momentum ---
    for p in [3, 6, 12, 24, 48]:
        df[f'mom_{p}'] = close / close.shift(p) - 1

    # --- Volatilite ---
    for window in [3, 6, 12, 24]:
        df[f'vol_{window}'] = close.pct_change().rolling(window).std()
        df[f'vol_ratio_{window}'] = df[f'vol_{window}'] / (df[f'vol_{window}'].shift(window) + 1e-8)

    # --- Hareketli ortalamalar ---
    for window in [5, 10, 20, 50]:
        ma = close.rolling(window).mean()
        df[f'ma_{window}'] = ma
        df[f'ma_ratio_{window}'] = close / (ma + 1e-8) - 1

    # --- MA crossover ---
    for short, long in [(5, 20), (10, 50), (3, 12)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)

    # --- Fiyat pozisyonu (Donchian kanalı) ---
    for window in [12, 24, 48]:
        roll_min = close.rolling(window).min()
        roll_max = close.rolling(window).max()
        df[f'price_pos_{window}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)

    # --- RSI ---
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    for window in [12, 20]:
        bb_mid = close.rolling(window).mean()
        bb_std = close.rolling(window).std()
        df[f'bb_pos_{window}'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-8)
        df[f'bb_width_{window}'] = (4 * bb_std) / (bb_mid + 1e-8)

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9).mean()
    df['macd'] = macd / (close + 1e-8)
    df['macd_signal'] = signal_line / (close + 1e-8)
    df['macd_hist'] = (macd - signal_line) / (close + 1e-8)

    # --- ATR (Average True Range) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean() / (close + 1e-8)

    # --- Stochastic ---
    for window in [14]:
        low_min = low.rolling(window).min()
        high_max = high.rolling(window).max()
        df[f'stoch_{window}'] = (close - low_min) / (high_max - low_min + 1e-8)

    # --- Zaman özellikleri ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_morning'] = ((df.index.hour >= 9) & (df.index.hour <= 12)).astype(int)
    df['is_afternoon'] = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)
    df['is_open_hour'] = (df.index.hour == 9).astype(int)
    df['is_close_hour'] = (df.index.hour == 17).astype(int)

    # --- Hacim özellikleri ---
    if 'Volume' in df.columns:
        vol = df['Volume']
        for window in [6, 12, 24]:
            df[f'vol_ma_{window}'] = vol.rolling(window).mean()
        df['vol_ratio'] = vol / (df['vol_ma_12'] + 1e-8)
        df['vol_spike'] = (vol > vol.rolling(24).mean() * 2).astype(int)
        df['price_vol'] = close.pct_change() * vol

    # --- Normalize fiyat ---
    for window in [12, 24, 48]:
        df[f'price_norm_{window}'] = (close - close.rolling(window).mean()) / (close.rolling(window).std() + 1e-8)

    # --- Streak ---
    ret = close.pct_change()
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


def get_feature_columns(df):
    """Özellik kolonlarını al (Close, High, Low, Open, Volume hariç)."""
    exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'}
    return [c for c in df.columns if c not in exclude]


# ============================================================
# HEDEF DEĞİŞKEN
# ============================================================
def create_target(df, horizon_bars):
    """horizon_bars sonraki kapanış bugünden yüksek mi?"""
    future_ret = df['Close'].pct_change(horizon_bars).shift(-horizon_bars)
    target = (future_ret > 0.0).astype(int)
    return target, future_ret


# ============================================================
# ÖZELLİK SEÇİMİ
# ============================================================
def select_features(X, y, k=35):
    """Mutual info + korelasyon filtresi."""
    from sklearn.feature_selection import mutual_info_classif

    valid_cols = X.columns[X.isna().mean() < 0.3].tolist()
    X_clean = X[valid_cols].fillna(X[valid_cols].median())

    mi_scores = mutual_info_classif(X_clean, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=valid_cols).sort_values(ascending=False)

    top_features = mi_series.head(k * 2).index.tolist()
    X_top = X_clean[top_features]

    corr_matrix = X_top.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    filtered = [f for f in top_features if f not in to_drop]

    selected = filtered[:k]
    print(f"    Özellik seçimi: {len(valid_cols)} → {len(top_features)} → {len(selected)}")
    return selected


# ============================================================
# MODEL EĞİTİMİ
# ============================================================
def train_model(X_train, y_train, X_test, y_test):
    """Stacking ensemble eğit."""
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-8)

    base_models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=10, gamma=0.3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
    }

    # Base modelleri eğit
    for name, m in base_models.items():
        m.fit(X_tr_sc, y_train)

    # OOF tahminleri (stacking için)
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(X_train), len(base_models)))

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_cv_tr = scaler.fit_transform(X_train.iloc[tr_idx])
        X_cv_val = scaler.transform(X_train.iloc[val_idx])
        for m_idx, (name, m) in enumerate(base_models.items()):
            m_fold = type(m)(**m.get_params())
            m_fold.fit(X_cv_tr, y_train.iloc[tr_idx])
            oof_preds[val_idx, m_idx] = m_fold.predict_proba(X_cv_val)[:, 1]

    # Meta-learner
    meta_learner = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    meta_learner.fit(oof_preds, y_train)

    # Test tahminleri
    test_base_preds = np.column_stack([
        m.predict_proba(X_te_sc)[:, 1] for m in base_models.values()
    ])
    stacking_proba = meta_learner.predict_proba(test_base_preds)[:, 1]
    avg_proba = np.mean(test_base_preds, axis=1)
    final_proba = stacking_proba * 0.6 + avg_proba * 0.4

    preds = (final_proba > 0.5).astype(int)
    test_acc = accuracy_score(y_test, preds)
    try:
        test_auc = roc_auc_score(y_test, final_proba)
    except:
        test_auc = 0.5

    # CV doğruluğu
    cv_accs = []
    cv_aucs = []
    tscv2 = TimeSeriesSplit(n_splits=5)
    for tr_idx, val_idx in tscv2.split(X_train):
        X_cv_tr = scaler.fit_transform(X_train.iloc[tr_idx])
        X_cv_val = scaler.transform(X_train.iloc[val_idx])
        cv_probas = []
        for name, m in base_models.items():
            m2 = type(m)(**m.get_params())
            m2.fit(X_cv_tr, y_train.iloc[tr_idx])
            cv_probas.append(m2.predict_proba(X_cv_val)[:, 1])
        cv_avg = np.mean(cv_probas, axis=0)
        cv_preds = (cv_avg > 0.5).astype(int)
        cv_accs.append(accuracy_score(y_train.iloc[val_idx], cv_preds))
        try:
            cv_aucs.append(roc_auc_score(y_train.iloc[val_idx], cv_avg))
        except:
            cv_aucs.append(0.5)

    return {
        'scaler': scaler,
        'base_models': base_models,
        'meta_learner': meta_learner,
        'test_accuracy': round(test_acc, 4),
        'test_auc': round(test_auc, 4),
        'cv_accuracy': round(np.mean(cv_accs), 4),
        'cv_std': round(np.std(cv_accs), 4),
        'cv_auc': round(np.mean(cv_aucs), 4),
        'positive_rate': round(float(y_train.mean()), 4),
        'train_size': len(X_train),
        'test_size': len(X_test),
    }


# ============================================================
# ANA EĞİTİM
# ============================================================
def run_hourly_training():
    print("\n" + "=" * 65)
    print("  GMSTR SAATLİK MODEL EĞİTİMİ (1h ve 4h)")
    print("=" * 65)
    print(f"  Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")

    # Veri yükle
    df = load_hourly_data()
    if df is None:
        print("❌ Saatlik veri alınamadı!")
        return False

    # Özellik mühendisliği
    df = engineer_hourly_features(df)
    feature_cols = get_feature_columns(df)
    print(f"[Özellik] {len(feature_cols)} özellik üretildi")

    # Saatlik horizon'lar: 1h = 1 bar, 4h = 4 bar
    horizons = {
        '1h': 1,
        '4h': 4,
    }

    all_results = {}

    for h_name, h_bars in horizons.items():
        print(f"\n{'─' * 65}")
        print(f"  VADE: {h_name} ({h_bars} bar sonrası)")
        print(f"{'─' * 65}")

        # Hedef oluştur
        target, future_ret = create_target(df, h_bars)

        clean = df.copy()
        clean['target'] = target
        clean['future_ret'] = future_ret
        clean = clean.dropna(subset=['target', 'future_ret'])

        # Geçerli feature kolonları
        valid_features = [c for c in feature_cols if c in clean.columns]
        clean_feat = clean.dropna(subset=valid_features, thresh=int(len(valid_features) * 0.7))

        if len(clean_feat) < 100:
            print(f"  ⚠ Yetersiz veri: {len(clean_feat)} satır")
            continue

        X = clean_feat[valid_features].fillna(clean_feat[valid_features].median())
        y = clean_feat['target']

        pos_rate = y.mean()
        print(f"  Veri: {len(clean_feat)} satır | Pozitif: {pos_rate:.1%} | Negatif: {1 - pos_rate:.1%}")

        # Özellik seçimi
        selected = select_features(X, y, k=35)
        X = X[selected]
        print(f"  Seçilen özellik: {len(selected)}")

        # Train/test split (son %20 test)
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"  Eğitim: {len(X_train)} | Test: {len(X_test)}")
        print(f"  Stacking ensemble eğitiliyor...")

        result = train_model(X_train, y_train, X_test, y_test)

        print(f"\n  ┌─ SONUÇLAR ─────────────────────────────────────┐")
        print(f"  │ CV Doğruluk:   {result['cv_accuracy']:.1%} ± {result['cv_std']:.1%}")
        print(f"  │ CV AUC:        {result['cv_auc']:.3f}")
        print(f"  │ Test Doğruluk: {result['test_accuracy']:.1%}")
        print(f"  │ Test AUC:      {result['test_auc']:.3f}")

        # Gerçekçilik kontrolü
        is_realistic = result['test_accuracy'] >= 0.50 and result['test_auc'] >= 0.50
        status = "✅ SAĞLIKLI" if is_realistic else "⚠️ ZAYIF"
        print(f"  │ Durum:         {status}")
        print(f"  └────────────────────────────────────────────────┘")

        # Modeli kaydet
        key = f'{h_name}_hourly'
        model_data = {
            'scaler': result['scaler'],
            'base_models': result['base_models'],
            'meta_learner': result['meta_learner'],
            'feature_cols': selected,
            'horizon': h_name,
            'frequency': 'hourly',
            'trained_at': datetime.now().isoformat(),
            'test_accuracy': result['test_accuracy'],
            'test_auc': result['test_auc'],
        }

        pkl_path = MODEL_DIR / f'simple_{h_name}_hourly.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  💾 Model kaydedildi: {pkl_path.name}")

        all_results[key] = {
            'horizon': h_name,
            'frequency': 'hourly',
            'cv_accuracy': result['cv_accuracy'],
            'cv_std': result['cv_std'],
            'cv_auc': result['cv_auc'],
            'test_accuracy': result['test_accuracy'],
            'test_auc': result['test_auc'],
            'positive_rate': result['positive_rate'],
            'train_size': result['train_size'],
            'test_size': result['test_size'],
            'is_realistic': is_realistic,
            'issues': [] if is_realistic else ['⚠️ Düşük doğruluk'],
        }

    # training_results.json'a ekle (mevcut günlük modelleri koru)
    results_path = MODEL_DIR / 'training_results.json'
    existing = {}
    if results_path.exists():
        try:
            with open(results_path, encoding='utf-8') as f:
                existing = json.load(f)
        except:
            pass

    existing.update(all_results)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n{'=' * 65}")
    print("  SAATLİK EĞİTİM TAMAMLANDI")
    print(f"{'=' * 65}")
    print(f"  {len(all_results)} saatlik model eğitildi")
    for key, r in all_results.items():
        print(f"  • {key}: Test={r['test_accuracy']:.1%} | AUC={r['test_auc']:.3f} | {'✅' if r['is_realistic'] else '⚠️'}")
    print(f"\n  Sonuçlar: {results_path}")
    print(f"{'=' * 65}\n")

    return len(all_results) > 0


if __name__ == '__main__':
    success = run_hourly_training()
    if success:
        print("✅ Saatlik model eğitimi tamamlandı!")
        print("   Şimdi tahminleri güncellemek için: python generate_predictions.py")
    else:
        print("❌ Eğitim başarısız!")
        sys.exit(1)
