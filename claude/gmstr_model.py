#!/usr/bin/env python3
"""
GMSTR (QNB Finans Portföy Gümüş BYF) Fiyat Tahmin Modeli
========================================================
Tahmin Ufukları: 1 saat | 4 saat | 1 gün | 1 hafta
Hedef: %60+ yön doğruluğu
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json, os, sys

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report, 
                              confusion_matrix, roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import ta
import joblib
import yfinance as yf

df = yf.download("GMSTR.IS", period="5y", interval="1d")
df.to_csv("gercek_data_5y_1d.csv")
df = pd.read_csv("gercek_data_5y_1d.csv")
print(df.head(10))
print("\nSütunlar:", df.columns.tolist())
print("\nTipler:", df.dtypes)

# ================================================================
# CONFIG
# ================================================================
HORIZONS = {
    '1d':  1,   # 1 gün  = 1 bar
    '5d':  5,   # 5 gün  = 1 hafta
    '22d': 22,  # 22 gün = ~1 ay
    '66d': 66,  # 66 gün = ~3 ay
}
THRESHOLD = 0.001  # Min fiyat hareketi yön sayılması için (%0.1)
MODEL_DIR = '/home/claude/gmstr_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================================================
# 1. VERİ OLUŞTURMA (Gerçek veri CSV ile değiştirilebilir)
# ================================================================

def generate_gmstr_data(n_days=1250, seed=42):
    """BIST gümüş ETF benzeri günlük veri üret"""
    np.random.seed(seed)
    
    # Sadece hafta içi günler
    start = datetime(2021, 1, 4)
    all_dates = pd.bdate_range(start=start, periods=n_days)
    
    n = len(all_dates)
    start_price = 48.0
    prices = [start_price]
    volumes = []
    
    regime = 'neutral'
    regime_counter = 0
    
    params = {
        'bull':     {'mu': 0.0025, 'sigma': 0.018},
        'bear':     {'mu': -0.0020, 'sigma': 0.022},
        'neutral':  {'mu': 0.0006, 'sigma': 0.012},
        'volatile': {'mu': 0.0002, 'sigma': 0.035},
    }
    
    for i in range(1, n):
        regime_counter += 1
        if regime_counter > np.random.randint(12, 45):
            regime = np.random.choice(['bull', 'bear', 'neutral', 'volatile'],
                                       p=[0.30, 0.22, 0.35, 0.13])
            regime_counter = 0
        
        shock = np.random.normal(params[regime]['mu'], params[regime]['sigma'])
        if np.random.random() < 0.003:
            shock += np.random.choice([-1, 1]) * np.random.uniform(0.04, 0.10)
        
        prices.append(prices[-1] * np.exp(shock))
        base_vol = 300000 + 200000 * abs(shock) / params[regime]['sigma']
        volumes.append(int(base_vol * np.random.lognormal(0, 0.5)))
    
    volumes.append(volumes[-1] if volumes else 300000)
    
    opens, highs, lows, closes = [], [], [], []
    for p in prices:
        noise = p * 0.008
        o = p + np.random.uniform(-noise, noise)
        h = p + abs(np.random.normal(0, noise * 1.5))
        l = p - abs(np.random.normal(0, noise * 1.5))
        opens.append(o)
        highs.append(max(o, h, p))
        lows.append(min(o, l, p))
        closes.append(p)
    
    df = pd.DataFrame({
        'Open': opens, 'High': highs, 'Low': lows,
        'Close': closes, 'Volume': volumes
    }, index=pd.DatetimeIndex(all_dates))
    
    return df

def load_real_data(csv_path):
    """Gerçek CSV verisi yükle - yfinance tüm formatları destekler"""
    # Ham oku, header yok
    raw = pd.read_csv(csv_path, header=None)
    
    # CSV yapısı:
    # Satır 0: Header (Price, Close, High, Low, Open, Volume)
    # Satır 1: Ticker bilgisi (GMSTR.IS)
    # Satır 2: Boş Date satırı
    # Satır 3+: Gerçek veri
    
    header_row = raw.iloc[0].tolist()  # ['Price','Close','High','Low','Open','Volume']
    
    # Veri satırlarını al (ilk 3 satır header/ticker/boş date)
    data = raw.iloc[3:].copy()
    data.columns = header_row
    data = data.rename(columns={'Price': 'Date'})
    data = data.set_index('Date')
    
    # Sayısala çevir
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Index'i datetime'a çevir
    data.index = pd.to_datetime(data.index, errors='coerce')
    data = data[data.index.notna()]
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data = data.sort_index()
    
    print(f"Yüklendi: {len(data)} satır | {data['Close'].iloc[0]:.2f} → {data['Close'].iloc[-1]:.2f} TRY")
    return data

# ================================================================
# 2. TEKNİK İNDİKATÖRLER + ÖZELLİK MÜHENDİSLİĞİ
# ================================================================

def add_technical_indicators(df):
    """Kapsamlı teknik analiz özellikleri"""
    df = df.copy()
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']
    
    # --- Momentum ---
    df['rsi_14']  = ta.momentum.RSIIndicator(c, 14).rsi()
    df['rsi_7']   = ta.momentum.RSIIndicator(c, 7).rsi()
    df['rsi_21']  = ta.momentum.RSIIndicator(c, 21).rsi()
    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['tsi']     = ta.momentum.TSIIndicator(c).tsi()
    df['uo']      = ta.momentum.UltimateOscillator(h, l, c).ultimate_oscillator()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
    
    # --- Trend ---
    macd = ta.trend.MACD(c)
    df['macd']        = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff']   = macd.macd_diff()
    
    for w in [5, 10, 20, 50]:
        df[f'ema_{w}'] = ta.trend.EMAIndicator(c, w).ema_indicator()
        df[f'sma_{w}'] = ta.trend.SMAIndicator(c, w).sma_indicator()
    
    df['adx']      = ta.trend.ADXIndicator(h, l, c, 14).adx()
    df['adx_pos']  = ta.trend.ADXIndicator(h, l, c, 14).adx_pos()
    df['adx_neg']  = ta.trend.ADXIndicator(h, l, c, 14).adx_neg()
    df['cci']      = ta.trend.CCIIndicator(h, l, c, 20).cci()
    df['dpo']      = ta.trend.DPOIndicator(c, 20).dpo()
    
    # Ichimoku (uzun vadeli)
    ich = ta.trend.IchimokuIndicator(h, l, 9, 26, 52)
    df['ich_a'] = ich.ichimoku_a()
    df['ich_b'] = ich.ichimoku_b()
    df['ich_base'] = ich.ichimoku_base_line()
    
    # --- Volatilite ---
    bb = ta.volatility.BollingerBands(c, 20, 2)
    df['bb_high']  = bb.bollinger_hband()
    df['bb_low']   = bb.bollinger_lband()
    df['bb_mid']   = bb.bollinger_mavg()
    df['bb_width'] = bb.bollinger_wband()
    df['bb_pct']   = bb.bollinger_pband()
    
    df['atr']  = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df['natr'] = df['atr'] / c  # Normalize ATR
    df['kc_high'] = ta.volatility.KeltnerChannel(h, l, c).keltner_channel_hband()
    df['kc_low']  = ta.volatility.KeltnerChannel(h, l, c).keltner_channel_lband()
    df['dc_high'] = ta.volatility.DonchianChannel(h, l, c).donchian_channel_hband()
    df['dc_low']  = ta.volatility.DonchianChannel(h, l, c).donchian_channel_lband()
    
    # --- Hacim ---
    df['obv']  = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v, 14).volume_weighted_average_price()
    df['mfi']  = ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index()
    df['cmf']  = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow()
    df['fi']   = ta.volume.ForceIndexIndicator(c, v, 13).force_index()
    df['eom']  = ta.volume.EaseOfMovementIndicator(h, l, v, 14).ease_of_movement()
    
    # --- Türetilmiş Özellikler ---
    # Fiyat göreceli hareketler
    for lag in [1, 2, 3, 4, 6, 8, 12, 24, 40]:
        df[f'ret_{lag}'] = c.pct_change(lag)
    
    # Fiyat pozisyonu
    df['hl_pct']  = (c - l) / (h - l + 1e-8)
    df['price_vs_ema20'] = (c - df['ema_20']) / df['ema_20']
    df['price_vs_ema50'] = (c - df['ema_50']) / df['ema_50']
    df['price_vs_vwap']  = (c - df['vwap']) / df['vwap']
    
    # EMA crossover sinyalleri
    df['ema5_cross_ema20']  = df['ema_5'] - df['ema_20']
    df['ema10_cross_ema50'] = df['ema_10'] - df['ema_50']
    
    # RSI divergence proxy
    df['rsi_diff']  = df['rsi_14'].diff(4)
    df['price_diff'] = c.pct_change(4)
    
    # Hacim teyidi
    df['vol_ratio']   = v / v.rolling(20).mean()
    df['vol_trend']   = v.rolling(5).mean() / v.rolling(20).mean()
    
    # Zaman özellikleri (günlük veri)
    df['day_of_week']  = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month']        = df.index.month
    df['quarter']      = df.index.quarter
    df['dow_sin']      = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['dow_cos']      = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['dom_sin']      = np.sin(2 * np.pi * df.index.day / 31)
    df['dom_cos']      = np.cos(2 * np.pi * df.index.day / 31)
    df['month_sin']    = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos']    = np.cos(2 * np.pi * df.index.month / 12)
    
    # Realized volatility
    df['rv_8']  = df['ret_1'].rolling(8).std() * np.sqrt(252 * 8)
    df['rv_24'] = df['ret_1'].rolling(24).std() * np.sqrt(252 * 8)
    df['rv_40'] = df['ret_1'].rolling(40).std() * np.sqrt(252 * 8)
    
    # Momentum score
    df['mom_score'] = (
        (df['rsi_14'] - 50) / 50 * 0.3 +
        np.sign(df['macd_diff']) * 0.2 +
        np.sign(df['ema5_cross_ema20']) * 0.2 +
        (df['stoch_k'] - 50) / 50 * 0.15 +
        (df['cci'] / 200).clip(-1, 1) * 0.15
    )
    
    return df

def create_targets(df, horizons=HORIZONS, threshold=THRESHOLD):
    """Her tahmin ufku için hedef etiket oluştur (1=yükseliş, 0=düşüş)"""
    for name, bars in horizons.items():
        future_ret = df['Close'].pct_change(bars).shift(-bars)
        df[f'target_{name}'] = (future_ret > threshold).astype(int)
        df[f'future_ret_{name}'] = future_ret
    return df

# ================================================================
# 3. ENSEMBLE MODEL (XGBoost + LightGBM + RF + GBM)
# ================================================================

def build_ensemble(n_estimators=200):
    """Güçlü ensemble model oluştur"""
    xgb_clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=5, learning_rate=0.05,
        num_leaves=31, subsample=0.8,
        colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    rf_clf = RandomForestClassifier(
        n_estimators=150, max_depth=8,
        min_samples_leaf=5, max_features='sqrt',
        random_state=42, n_jobs=-1
    )
    gb_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4,
        learning_rate=0.08, subsample=0.8,
        random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('lgb', lgb_clf),
            ('rf',  rf_clf),
            ('gb',  gb_clf),
        ],
        voting='soft',
        weights=[3, 3, 2, 2]
    )
    return ensemble

# ================================================================
# 4. WALK-FORWARD VALIDASYON (Gerçekçi backtesting)
# ================================================================

def walk_forward_validate(X, y, n_splits=5):
    """Zaman serisi walk-forward validasyonu"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    fold_aucs = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        
        model = build_ensemble(n_estimators=100)
        model.fit(X_train_s, y_train)
        
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        
        fold_scores.append(acc)
        fold_aucs.append(auc)
    
    return {
        'cv_accuracy_mean': np.mean(fold_scores),
        'cv_accuracy_std':  np.std(fold_scores),
        'cv_auc_mean':      np.mean(fold_aucs),
        'cv_auc_std':       np.std(fold_aucs),
        'fold_scores':      fold_scores,
    }

# ================================================================
# 5. ANA EĞİTİM & DEĞERLENDİRME
# ================================================================

def train_and_evaluate(df_raw, verbose=True):
    """Ana eğitim + güvenilirlik testi"""
    
    if verbose:
        print("\n" + "="*60)
        print("GMSTR TAHMİN MODELİ EĞİTİMİ")
        print("="*60)
    
    # Feature engineering
    if verbose: print("\n[1/5] Teknik indikatörler hesaplanıyor...")
    df = add_technical_indicators(df_raw)
    df = create_targets(df)
    
    # Feature columns
    feature_cols = [c for c in df.columns if c not in 
                    ['Open','High','Low','Close','Volume','regime',
                     'day_of_week','day_of_month','month','quarter'] 
                    and not c.startswith('target_') 
                    and not c.startswith('future_ret_')]
    
    results = {}
    trained_models = {}
    
    for horizon_name in HORIZONS.keys():
        target_col = f'target_{horizon_name}'
        
        # Clean data
        df_clean = df.dropna(subset=feature_cols + [target_col])
        # Sonsuz ve çok büyük değerleri temizle
        df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna(subset=feature_cols)
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Train/test split (80/20, temporal)
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if verbose:
            print(f"\n[Horizon: {horizon_name}]")
            print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
            print(f"  Label balance: {y_train.mean():.2%} pozitif")
        
        # Scaler
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        
        # Walk-forward CV
        if verbose: print(f"  Walk-forward validasyon yapılıyor...")
        cv_results = walk_forward_validate(X_train, y_train, n_splits=5)
        
        # Final model eğitimi
        if verbose: print(f"  Final model eğitiliyor...")
        final_model = build_ensemble(n_estimators=200)
        final_model.fit(X_train_s, y_train)
        
        # Test değerlendirmesi
        preds_test  = final_model.predict(X_test_s)
        proba_test  = final_model.predict_proba(X_test_s)[:, 1]
        test_acc    = accuracy_score(y_test, preds_test)
        test_auc    = roc_auc_score(y_test, proba_test)
        
        # Yüksek güven tahminleri (proba > 0.65 veya < 0.35)
        high_conf_mask = (proba_test > 0.65) | (proba_test < 0.35)
        high_conf_acc  = accuracy_score(
            y_test[high_conf_mask], preds_test[high_conf_mask]
        ) if high_conf_mask.sum() > 10 else None
        high_conf_ratio = high_conf_mask.mean()
        
        # Feature importance (XGBoost)
        xgb_model = final_model.estimators_[0]  # XGBoost
        feat_imp = pd.Series(
            xgb_model.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False).head(10).to_dict()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, preds_test).tolist()
        
        result = {
            'horizon':           horizon_name,
            'train_size':        len(X_train),
            'test_size':         len(X_test),
            'cv_acc_mean':       round(cv_results['cv_accuracy_mean'], 4),
            'cv_acc_std':        round(cv_results['cv_accuracy_std'], 4),
            'cv_auc_mean':       round(cv_results['cv_auc_mean'], 4),
            'test_accuracy':     round(test_acc, 4),
            'test_auc':          round(test_auc, 4),
            'high_conf_accuracy':round(high_conf_acc, 4) if high_conf_acc else None,
            'high_conf_ratio':   round(high_conf_ratio, 4),
            'top_features':      {k: round(float(v), 4) for k, v in feat_imp.items()},
            'confusion_matrix':  cm,
            'fold_scores':       [round(f, 4) for f in cv_results['fold_scores']],
        }
        
        results[horizon_name] = result
        trained_models[horizon_name] = {
            'model': final_model,
            'scaler': scaler,
            'features': feature_cols,
        }
        
        if verbose:
            print(f"\n  ━━━ SONUÇLAR ({horizon_name}) ━━━")
            print(f"  CV Doğruluk:       {result['cv_acc_mean']:.2%} ± {result['cv_acc_std']:.2%}")
            print(f"  Test Doğruluğu:    {result['test_accuracy']:.2%}")
            print(f"  Test AUC-ROC:      {result['test_auc']:.4f}")
            if high_conf_acc:
                print(f"  Yüksek Güven Acc:  {result['high_conf_accuracy']:.2%} ({result['high_conf_ratio']:.0%} veri)")
    
    return results, trained_models, feature_cols

# ================================================================
# 6. TAHMİN FONKSİYONU
# ================================================================

def predict_now(df_raw, trained_models, feature_cols, last_n=200):
    """Mevcut son veriden tahmin üret"""
    df = add_technical_indicators(df_raw.tail(max(last_n, 200)))
    df_clean = df.dropna(subset=feature_cols)
    
    if len(df_clean) == 0:
        return None
    
    last_row = df_clean.iloc[[-1]]
    X_last = last_row[feature_cols]
    current_price = df_raw['Close'].iloc[-1]
    current_time  = df_raw.index[-1]
    
    predictions = {}
    for horizon_name, bars in HORIZONS.items():
        model_data = trained_models[horizon_name]
        X_scaled = model_data['scaler'].transform(X_last)
        
        proba = model_data['model'].predict_proba(X_scaled)[0]
        direction = 'YUKARI ↑' if proba[1] > 0.5 else 'AŞAĞI ↓'
        confidence = max(proba)
        
        # Fiyat tahmini (proba ağırlıklı volatility-based estimate)
        cv_vol = df['ret_1'].std() * np.sqrt(bars)
        expected_move = cv_vol * (proba[1] - 0.5) * 2
        predicted_price = current_price * (1 + expected_move)
        
        predictions[horizon_name] = {
            'direction':        direction,
            'confidence':       round(confidence, 4),
            'prob_up':          round(proba[1], 4),
            'prob_down':        round(proba[0], 4),
            'current_price':    round(current_price, 2),
            'predicted_price':  round(predicted_price, 2),
            'expected_change_pct': round(expected_move * 100, 2),
        }
    
    return predictions, current_price, current_time

# ================================================================
# 7. MODEL KAYDET / YÜKLE
# ================================================================

def save_models(trained_models, feature_cols, results, model_dir=MODEL_DIR):
    for name, data in trained_models.items():
        joblib.dump(data['model'],  f"{model_dir}/model_{name}.pkl")
        joblib.dump(data['scaler'], f"{model_dir}/scaler_{name}.pkl")
    
    with open(f"{model_dir}/feature_cols.json", 'w') as f:
        json.dump(feature_cols, f)
    with open(f"{model_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModeller kaydedildi: {model_dir}/")

def load_models(model_dir=MODEL_DIR):
    trained_models = {}
    for name in HORIZONS.keys():
        trained_models[name] = {
            'model':  joblib.load(f"{model_dir}/model_{name}.pkl"),
            'scaler': joblib.load(f"{model_dir}/scaler_{name}.pkl"),
        }
    with open(f"{model_dir}/feature_cols.json") as f:
        feature_cols = json.load(f)
    return trained_models, feature_cols

# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    import sys
    
    # Gerçek veri CSV opsiyonu
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        print(f"Gerçek veri yükleniyor: {sys.argv[1]}")
        df_raw = load_real_data(sys.argv[1])
    else:
        print("Sentetik GMSTR verisi oluşturuluyor (1250 gün, günlük)...")
        df_raw = generate_gmstr_data(n_days=1250)
    
    print(f"Veri: {len(df_raw)} satır | {df_raw.index[0]} → {df_raw.index[-1]}")
    
    # Eğitim
    results, trained_models, feature_cols = train_and_evaluate(df_raw, verbose=True)
    
    # Kaydet
    save_models(trained_models, feature_cols, results)
    
    # Son tahmin
    print("\n" + "="*60)
    print("GÜNCEL TAHMİNLER")
    print("="*60)
    preds, cur_price, cur_time = predict_now(df_raw, trained_models, feature_cols)
    
    print(f"\nSon Fiyat: {cur_price:.2f} TRY  [{cur_time}]")
    print("-"*60)
    for h, p in preds.items():
        signal = "🟢" if "YUKARI" in p['direction'] else "🔴"
        conf_bar = "█" * int(p['confidence']*10)
        print(f"{signal} {h:>4}  {p['direction']:<12}  "
              f"Güven: {p['confidence']:.0%}  {conf_bar:<10}  "
              f"↗ {p['predicted_price']:.2f} TRY  ({p['expected_change_pct']:+.2f}%)")
    
    # Özet rapor
    print("\n" + "="*60)
    print("MODEL PERFORMANS ÖZETİ")
    print("="*60)
    print(f"{'Ufuk':<6} {'CV Doğruluk':<15} {'Test Doğruluk':<16} {'AUC':<8} {'YüksekGüven'}")
    print("-"*60)
    for h, r in results.items():
        hc = f"{r['high_conf_accuracy']:.0%} ({r['high_conf_ratio']:.0%})" if r['high_conf_accuracy'] else "N/A"
        print(f"{h:<6} {r['cv_acc_mean']:.2%} ± {r['cv_acc_std']:.2%}   "
              f"{r['test_accuracy']:.2%}          {r['test_auc']:.3f}   {hc}")
    
    print("\nModel dosyaları:", MODEL_DIR)
    print("\nKullanım: python3 gmstr_model.py [veri.csv]")
    
    # JSON çıktı
    with open('/home/claude/gmstr_results.json', 'w') as f:
        all_output = {
            'model_results': results,
            'latest_predictions': preds,
            'current_price': cur_price,
            'timestamp': str(cur_time)
        }
        json.dump(all_output, f, indent=2)
    print("\nSonuçlar: /home/claude/gmstr_results.json")
