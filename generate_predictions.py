"""
Yeni eğitilmiş modellerle GMSTR tahmini üret ve latest_predictions.json'a kaydet.
Günlük (1d/3d/5d/10d) + Saatlik (1h/4h) modeller desteklenir.
"""
import sys
import json
import pickle
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / 'gmstr_models'


# ============================================================
# CANLI FİYAT
# ============================================================
def get_live_price():
    """Yahoo Finance'den canlı GMSTR fiyatı çek."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GMSTR.IS")
        hist = ticker.history(period="1d", interval="1m")
        if len(hist) > 0:
            price = float(hist['Close'].iloc[-1])
            print(f"  [Canlı Fiyat] {price:.2f} TL (Yahoo Finance)")
            return price
    except Exception as e:
        print(f"  [Canlı Fiyat] Yahoo Finance hatası: {e}")

    # Fallback: CSV'den son fiyat
    try:
        csv_path = ROOT / 'claude' / 'areaxdatetime.csv'
        df = pd.read_csv(csv_path, encoding='utf-8')
        if 'Close' in df.columns:
            price = float(df['Close'].iloc[-1])
            print(f"  [Canlı Fiyat] {price:.2f} TL (CSV fallback)")
            return price
    except Exception as e:
        print(f"  [Canlı Fiyat] CSV hatası: {e}")

    return None


# ============================================================
# GÜNLÜK VERİ
# ============================================================
def load_daily_data():
    csv_path = ROOT / 'claude' / 'areaxdatetime.csv'
    from gmstr_system.data_loader import GMSTRDataLoader
    loader = GMSTRDataLoader(str(csv_path))
    loader.load()
    df = loader.clean()
    return df


def engineer_features(df):
    from gmstr_system.features import FeatureEngineer
    eng = FeatureEngineer()
    df = eng.transform(df)
    close = df['Close']

    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'ret_lag_{lag}'] = close.pct_change(lag)
        df[f'close_lag_{lag}'] = close.shift(lag)

    df['mom_5'] = close / close.shift(5) - 1
    df['mom_10'] = close / close.shift(10) - 1
    df['mom_20'] = close / close.shift(20) - 1
    df['mom_60'] = close / close.shift(60) - 1

    for window in [5, 10, 20]:
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

    if 'Volume' in df.columns:
        vol = df['Volume']
        df['vol_ma_5'] = vol.rolling(5).mean()
        df['vol_ma_20'] = vol.rolling(20).mean()
        df['vol_ratio'] = vol / (df['vol_ma_20'] + 1e-8)
        df['price_vol'] = close * vol

    for short, long in [(5, 20), (10, 50), (20, 60)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)

    for window in [10, 20]:
        df[f'channel_pos_{window}'] = (
            (close - close.rolling(window).min()) /
            (close.rolling(window).max() - close.rolling(window).min() + 1e-8)
        )

    daily_ret = close.pct_change()
    df['up_streak'] = (daily_ret > 0).astype(int)
    df['down_streak'] = (daily_ret < 0).astype(int)

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

    return df


# ============================================================
# SAATLİK VERİ
# ============================================================
def load_hourly_data():
    """Yahoo Finance'den saatlik GMSTR verisi çek."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("GMSTR.IS")
        # Son 60 günlük saatlik veri
        df = ticker.history(period="60d", interval="1h")
        if len(df) < 50:
            print(f"  [Saatlik Veri] Yetersiz veri: {len(df)} satır")
            return None
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        print(f"  [Saatlik Veri] {len(df)} satır yüklendi")
        return df
    except Exception as e:
        print(f"  [Saatlik Veri] Hata: {e}")
        return None


def engineer_hourly_features(df):
    """Saatlik veri için özellik mühendisliği."""
    close = df['Close'].copy()

    # Lag özellikleri
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'ret_lag_{lag}'] = close.pct_change(lag)
        df[f'close_lag_{lag}'] = close.shift(lag)

    # Momentum
    df['mom_3'] = close / close.shift(3) - 1
    df['mom_6'] = close / close.shift(6) - 1
    df['mom_12'] = close / close.shift(12) - 1
    df['mom_24'] = close / close.shift(24) - 1

    # Volatilite
    for window in [3, 6, 12, 24]:
        df[f'vol_{window}'] = close.pct_change().rolling(window).std()

    # Hareketli ortalamalar
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = close.rolling(window).mean()
        df[f'ma_ratio_{window}'] = close / (df[f'ma_{window}'] + 1e-8) - 1

    # MA crossover
    for short, long in [(5, 20), (10, 50)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)

    # Fiyat pozisyonu
    for window in [12, 24, 48]:
        roll_min = close.rolling(window).min()
        roll_max = close.rolling(window).max()
        df[f'price_pos_{window}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)

    # Zaman özellikleri
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_morning'] = ((df.index.hour >= 9) & (df.index.hour <= 12)).astype(int)
    df['is_afternoon'] = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(int)

    # Hacim özellikleri
    if 'Volume' in df.columns:
        vol = df['Volume']
        df['vol_ma_12'] = vol.rolling(12).mean()
        df['vol_ratio'] = vol / (df['vol_ma_12'] + 1e-8)

    # RSI (14 periyot)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = (bb_mid + 2 * bb_std - close) / (close + 1e-8)
    df['bb_lower'] = (close - (bb_mid - 2 * bb_std)) / (close + 1e-8)
    df['bb_pos'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-8)

    # Normalize fiyat
    df['price_norm_12'] = (close - close.rolling(12).mean()) / (close.rolling(12).std() + 1e-8)
    df['price_norm_24'] = (close - close.rolling(24).mean()) / (close.rolling(24).std() + 1e-8)

    return df


# ============================================================
# TAHMİN FONKSİYONU
# ============================================================
def predict_with_model(model_data, df):
    scaler = model_data['scaler']
    base_models = model_data['base_models']
    meta_learner = model_data['meta_learner']
    feature_cols = model_data['feature_cols']

    # Son satırı al
    valid_cols = [c for c in feature_cols if c in df.columns]
    last_row = df[valid_cols].iloc[-1:].fillna(df[valid_cols].median())

    # Eksik kolonları 0 ile doldur
    for col in feature_cols:
        if col not in last_row.columns:
            last_row[col] = 0.0

    last_row = last_row[feature_cols]
    X_sc = scaler.transform(last_row)

    # Base model tahminleri
    base_preds = np.column_stack([
        m.predict_proba(X_sc)[:, 1] for m in base_models.values()
    ])

    # Stacking
    stacking_proba = meta_learner.predict_proba(base_preds)[:, 1]
    avg_proba = np.mean(base_preds, axis=1)
    final_proba = float(stacking_proba[0] * 0.6 + avg_proba[0] * 0.4)

    return final_proba


def make_prediction_dict(prob_up, current_price, h_name, frequency, model_data, avg_move_per_period):
    """Tahmin sözlüğü oluştur (tüm HTML alanlarıyla)."""
    prob_down = 1 - prob_up

    # Sinyal - daha gerçekçi eşikler
    if prob_up > 0.60:
        signal = 'AL'
        signal_strength = 'GÜÇLÜ'
    elif prob_up > 0.53:
        signal = 'AL'
        signal_strength = 'ZAYIF'
    elif prob_up < 0.40:
        signal = 'SAT'
        signal_strength = 'GÜÇLÜ'
    elif prob_up < 0.47:
        signal = 'SAT'
        signal_strength = 'ZAYIF'
    else:
        signal = 'BEKLE'
        signal_strength = 'NÖTR'

    # Direction (HTML'in beklediği format)
    if prob_up > 0.5:
        direction = '\u2191 YUKARI'
    else:
        direction = '\u2193 A\u015eA\u011eI'

    # Güven skoru - iki farklı hesaplama:
    # 1. Ham güven: |prob_up - 0.5| * 2  (0-1 arası, modeller 0.5'e yakın olunca düşük)
    confidence_raw = abs(prob_up - 0.5) * 2  # 0-1 arası

    # 2. Normalize güven: prob_up'ı 50-100 arasına normalize et (kullanıcıya daha anlamlı)
    # prob_up=0.5 → 50%, prob_up=0.6 → 70%, prob_up=0.7 → 90%
    if prob_up >= 0.5:
        confidence_normalized = 50 + (prob_up - 0.5) * 100  # 50-100 arası
    else:
        confidence_normalized = 50 - (0.5 - prob_up) * 100  # 0-50 arası
    confidence_normalized = max(0, min(100, confidence_normalized))

    # Model doğruluğunu da hesaba kat
    model_acc = model_data.get('test_accuracy', 0.5)
    model_auc = model_data.get('test_auc', 0.5)
    
    # Kombine güven: prob_up normalize + model kalitesi
    model_quality_bonus = max(0, (model_acc - 0.5) * 2)  # 0-1 arası
    confidence_combined = confidence_normalized / 100 * (0.7 + 0.3 * model_quality_bonus)
    confidence_combined = max(0, min(1, confidence_combined))

    # Ana confidence değeri (HTML'de gösterilecek) - normalize edilmiş
    confidence = confidence_combined

    # Fiyat tahmini
    dir_mult = 1 if prob_up > 0.5 else -1
    magnitude = abs(prob_up - 0.5) * 2 * avg_move_per_period
    expected_change_pct = round(dir_mult * magnitude * 100, 2)
    forecast_price = current_price * (1 + dir_mult * magnitude) if current_price else None
    predicted_price = forecast_price  # HTML'de predicted_price olarak da kullanılıyor

    return {
        'horizon': h_name,
        'frequency': frequency,
        'prob_up': round(prob_up, 4),
        'prob_down': round(prob_down, 4),
        'signal': signal,
        'signal_strength': signal_strength,
        'direction': direction,                          # HTML için: ↑ YUKARI / ↓ AŞAĞI
        'confidence': round(confidence, 4),              # Kombine güven (0-1)
        'confidence_raw': round(confidence_raw, 4),      # Ham güven (|prob-0.5|*2)
        'confidence_pct': round(confidence_normalized, 1),  # Normalize % (50-100)
        'current_price': round(current_price, 2) if current_price else None,
        'forecast_price': round(forecast_price, 2) if forecast_price else None,
        'predicted_price': round(predicted_price, 2) if predicted_price else None,  # HTML alias
        'expected_change_pct': expected_change_pct,     # HTML için
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_accuracy': model_data.get('test_accuracy', 0),
        'model_auc': model_data.get('test_auc', 0),
        'trained_at': model_data.get('trained_at', ''),
    }


# ============================================================
# ANA FONKSİYON
# ============================================================
def main():
    print("=" * 60)
    print("  GMSTR TAHMİN ÜRETİMİ")
    print("=" * 60)

    # Canlı fiyat al
    live_price = get_live_price()

    predictions = {}

    # --------------------------------------------------------
    # GÜNLÜK TAHMİNLER (1d / 3d / 5d / 10d)
    # --------------------------------------------------------
    print("\n[Günlük Modeller]")
    try:
        df_daily = load_daily_data()
        df_daily = engineer_features(df_daily)
        csv_price = float(df_daily['Close'].iloc[-1])
        current_price = live_price if live_price else csv_price
        print(f"  Güncel fiyat: {current_price:.2f} TL")

        # 5d modeli KALDIRILDI (zararlı - %43.5 doğruluk)
        horizons = {'1d': (1, 0.015), '3d': (3, 0.015), '10d': (10, 0.015)}

        for h_name, (h_days, avg_move) in horizons.items():
            key = f'{h_name}_daily'
            pkl_path = MODEL_DIR / f'simple_{h_name}_daily.pkl'

            if not pkl_path.exists():
                print(f"  [{key}] Model bulunamadı: {pkl_path.name}")
                continue

            try:
                with open(pkl_path, 'rb') as f:
                    model_data = pickle.load(f)

                prob_up = predict_with_model(model_data, df_daily)
                pred = make_prediction_dict(prob_up, current_price, h_name, 'daily', model_data, avg_move * h_days)
                predictions[key] = pred
                print(f"  [{key}] prob_up={prob_up:.3f} | {pred['direction']} | {pred['signal']} ({pred['signal_strength']})")

            except Exception as e:
                print(f"  [{key}] HATA: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"  [Günlük] Veri yükleme hatası: {e}")

    # --------------------------------------------------------
    # SAATLİK TAHMİNLER (1h / 4h)
    # --------------------------------------------------------
    print("\n[Saatlik Modeller]")
    df_hourly = load_hourly_data()

    if df_hourly is not None:
        df_hourly = engineer_hourly_features(df_hourly)
        hourly_price = live_price if live_price else float(df_hourly['Close'].iloc[-1])

        hourly_horizons = {
            '1h': (1, 0.005),   # 1 saatlik, ortalama %0.5 hareket
            '4h': (4, 0.008),   # 4 saatlik, ortalama %0.8 hareket
        }

        for h_name, (h_periods, avg_move) in hourly_horizons.items():
            key = f'{h_name}_hourly'
            pkl_path = MODEL_DIR / f'simple_{h_name}_hourly.pkl'

            if not pkl_path.exists():
                print(f"  [{key}] Model bulunamadı: {pkl_path.name} (Önce eğitin!)")
                continue

            try:
                with open(pkl_path, 'rb') as f:
                    model_data = pickle.load(f)

                prob_up = predict_with_model(model_data, df_hourly)
                pred = make_prediction_dict(prob_up, hourly_price, h_name, 'hourly', model_data, avg_move * h_periods)
                predictions[key] = pred
                print(f"  [{key}] prob_up={prob_up:.3f} | {pred['direction']} | {pred['signal']} ({pred['signal_strength']})")

            except Exception as e:
                print(f"  [{key}] HATA: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("  Saatlik veri alınamadı, saatlik tahminler atlandı.")

    # --------------------------------------------------------
    # 15 DAKİKALIK TAHMİNLER
    # --------------------------------------------------------
    print("\n[15 Dakikalık Model]")
    pkl_15m = MODEL_DIR / 'simple_15m_15min.pkl'
    if pkl_15m.exists():
        try:
            import yfinance as yf
            ticker = yf.Ticker("GMSTR.IS")
            df_15m = ticker.history(period="60d", interval="15m")
            df_15m.index = pd.to_datetime(df_15m.index)
            if df_15m.index.tz is not None:
                df_15m.index = df_15m.index.tz_localize(None)
            df_15m = df_15m[(df_15m.index.hour >= 9) & (df_15m.index.hour <= 18)]
            df_15m = df_15m[df_15m.index.dayofweek < 5]

            if len(df_15m) >= 100:
                # 15m özellik mühendisliği (basit versiyon)
                close = df_15m['Close'].copy()
                ret = close.pct_change()
                for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48]:
                    df_15m[f'ret_{lag}'] = close.pct_change(lag)
                for p in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96]:
                    df_15m[f'mom_{p}'] = close / close.shift(p) - 1
                for w in [4, 8, 12, 16, 24, 32, 48]:
                    df_15m[f'vol_{w}'] = ret.rolling(w).std()
                df_15m['vol_ratio_4_24'] = df_15m['vol_4'] / (df_15m.get('vol_24', pd.Series(1, index=df_15m.index)) + 1e-8)
                df_15m['vol_ratio_8_48'] = df_15m['vol_8'] / (df_15m.get('vol_48', pd.Series(1, index=df_15m.index)) + 1e-8)
                for w in [4, 8, 12, 16, 20, 24, 32, 48, 64, 96]:
                    df_15m[f'ma_{w}'] = close.rolling(w).mean()
                    df_15m[f'ma_ratio_{w}'] = close / (df_15m[f'ma_{w}'] + 1e-8) - 1
                for w in [8, 16, 24, 48]:
                    df_15m[f'ema_{w}'] = close.ewm(span=w, adjust=False).mean()
                    df_15m[f'ema_ratio_{w}'] = close / (df_15m[f'ema_{w}'] + 1e-8) - 1
                for short, long in [(4, 16), (8, 24), (8, 48), (16, 48)]:
                    ma_s = close.rolling(short).mean()
                    ma_l = close.rolling(long).mean()
                    df_15m[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)
                    df_15m[f'ma_cross_sign_{short}_{long}'] = np.sign(ma_s - ma_l)
                for period in [7, 14, 21]:
                    delta = close.diff()
                    gain = delta.clip(lower=0).rolling(period).mean()
                    loss = (-delta.clip(upper=0)).rolling(period).mean()
                    rs = gain / (loss + 1e-8)
                    rsi = 100 - (100 / (1 + rs))
                    df_15m[f'rsi_{period}'] = rsi / 100
                    df_15m[f'rsi_{period}_ob'] = (rsi > 70).astype(int)
                    df_15m[f'rsi_{period}_os'] = (rsi < 30).astype(int)
                    df_15m[f'rsi_{period}_norm'] = (rsi - 50) / 50
                for fast, slow, sig in [(12, 26, 9), (8, 21, 8)]:
                    ema_f = close.ewm(span=fast, adjust=False).mean()
                    ema_s_m = close.ewm(span=slow, adjust=False).mean()
                    macd = ema_f - ema_s_m
                    signal = macd.ewm(span=sig, adjust=False).mean()
                    df_15m[f'macd_{fast}_{slow}'] = macd / (close + 1e-8)
                    df_15m[f'macd_hist_{fast}_{slow}'] = (macd - signal) / (close + 1e-8)
                    df_15m[f'macd_cross_{fast}_{slow}'] = np.sign(macd - signal)
                for w in [12, 20, 24]:
                    bb_mid = close.rolling(w).mean()
                    bb_std = close.rolling(w).std()
                    bb_upper = bb_mid + 2 * bb_std
                    bb_lower = bb_mid - 2 * bb_std
                    df_15m[f'bb_pos_{w}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    df_15m[f'bb_width_{w}'] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
                    df_15m[f'bb_above_{w}'] = (close > bb_upper).astype(int)
                    df_15m[f'bb_below_{w}'] = (close < bb_lower).astype(int)
                high = df_15m['High'].copy()
                low = df_15m['Low'].copy()
                open_ = df_15m['Open'].copy()
                volume = df_15m['Volume'].copy() if 'Volume' in df_15m.columns else pd.Series(1, index=df_15m.index)
                for w in [9, 14, 21]:
                    low_min = low.rolling(w).min()
                    high_max = high.rolling(w).max()
                    stoch_k = (close - low_min) / (high_max - low_min + 1e-8) * 100
                    df_15m[f'stoch_k_{w}'] = stoch_k / 100
                    df_15m[f'stoch_ob_{w}'] = (stoch_k > 80).astype(int)
                    df_15m[f'stoch_os_{w}'] = (stoch_k < 20).astype(int)
                tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                for w in [7, 14, 21]:
                    df_15m[f'atr_{w}'] = tr.rolling(w).mean() / (close + 1e-8)
                for w in [8, 16, 24, 48, 96]:
                    roll_min = close.rolling(w).min()
                    roll_max = close.rolling(w).max()
                    df_15m[f'price_pos_{w}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)
                    df_15m[f'dist_high_{w}'] = (roll_max - close) / (close + 1e-8)
                    df_15m[f'dist_low_{w}'] = (close - roll_min) / (close + 1e-8)
                for w in [8, 16, 24, 48]:
                    vol_ma = volume.rolling(w).mean()
                    df_15m[f'vol_ratio_v_{w}'] = volume / (vol_ma + 1e-8)
                obv = (np.sign(close.diff()) * volume).cumsum()
                for w in [16, 32]:
                    obv_ma = obv.rolling(w).mean()
                    df_15m[f'obv_trend_{w}'] = np.sign(obv - obv_ma)
                df_15m['candle_body'] = (close - open_) / (close + 1e-8)
                df_15m['candle_range'] = (high - low) / (close + 1e-8)
                df_15m['candle_is_bull'] = (close > open_).astype(int)
                df_15m['candle_upper_shadow'] = (high - close.clip(lower=open_)) / (close + 1e-8)
                df_15m['candle_lower_shadow'] = (close.clip(upper=open_) - low) / (close + 1e-8)
                df_15m['hour'] = df_15m.index.hour
                df_15m['minute'] = df_15m.index.minute
                df_15m['day_of_week'] = df_15m.index.dayofweek
                df_15m['hour_sin'] = np.sin(2 * np.pi * df_15m.index.hour / 24)
                df_15m['hour_cos'] = np.cos(2 * np.pi * df_15m.index.hour / 24)
                df_15m['is_morning_open'] = ((df_15m.index.hour == 9) & (df_15m.index.minute <= 30)).astype(int)
                df_15m['is_morning'] = ((df_15m.index.hour >= 9) & (df_15m.index.hour <= 11)).astype(int)
                df_15m['is_midday'] = ((df_15m.index.hour >= 11) & (df_15m.index.hour <= 14)).astype(int)
                df_15m['is_afternoon'] = ((df_15m.index.hour >= 14) & (df_15m.index.hour <= 17)).astype(int)
                df_15m['is_close'] = (df_15m.index.hour >= 17).astype(int)
                df_15m['is_monday'] = (df_15m.index.dayofweek == 0).astype(int)
                df_15m['is_friday'] = (df_15m.index.dayofweek == 4).astype(int)
                for w in [16, 24, 48]:
                    df_15m[f'price_norm_{w}'] = (close - close.rolling(w).mean()) / (close.rolling(w).std() + 1e-8)
                plus_dm = (high - high.shift(1)).clip(lower=0)
                minus_dm = (low.shift(1) - low).clip(lower=0)
                atr14 = tr.rolling(14).mean()
                plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-8)
                minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-8)
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
                adx = dx.rolling(14).mean()
                df_15m['adx'] = adx / 100
                df_15m['di_diff'] = (plus_di - minus_di) / 100
                df_15m['is_trending'] = (adx > 25).astype(int)
                streak = []
                current_s = 0
                for r in ret:
                    if pd.isna(r):
                        streak.append(0)
                    elif r > 0:
                        current_s = max(0, current_s) + 1
                        streak.append(current_s)
                    elif r < 0:
                        current_s = min(0, current_s) - 1
                        streak.append(current_s)
                    else:
                        current_s = 0
                        streak.append(0)
                df_15m['price_streak'] = streak

                price_15m = live_price if live_price else float(df_15m['Close'].iloc[-1])
                with open(pkl_15m, 'rb') as f:
                    model_data_15m = pickle.load(f)
                prob_up_15m = predict_with_model(model_data_15m, df_15m)
                pred_15m = make_prediction_dict(prob_up_15m, price_15m, '15m', '15min', model_data_15m, 0.003)
                predictions['15m_15min'] = pred_15m
                print(f"  [15m_15min] prob_up={prob_up_15m:.3f} | {pred_15m['direction']} | {pred_15m['signal']} ({pred_15m['signal_strength']})")
            else:
                print(f"  [15m] Yetersiz veri: {len(df_15m)} bar")
        except Exception as e:
            print(f"  [15m] HATA: {e}")
    else:
        print("  [15m] Model bulunamadı. Önce: python train_15m_model.py")

    # --------------------------------------------------------
    # KAYDET
    # --------------------------------------------------------
    pred_path = MODEL_DIR / 'latest_predictions.json'
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n{len(predictions)} tahmin kaydedildi: {pred_path}")
    for key, p in predictions.items():
        print(f"  {key}: {p['direction']} | prob_up={p['prob_up']} | fiyat={p['current_price']} TL")
    print("Tamamlandı!")


if __name__ == '__main__':
    main()
