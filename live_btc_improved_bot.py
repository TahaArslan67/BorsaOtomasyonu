"""
BTC Geliştirilmiş Canlı Yön Tahmin Botu
Fiyat Tahminli
"""

import numpy as np
import pandas as pd
import logging
import time
from exchange_client_crypto import CryptoExchangeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI hesapla"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR hesapla"""
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    return df['tr'].rolling(window=window).mean()


def calculate_z_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Fiyatın Hareketli Ortalamadan Sapması (Z-Score)"""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    z_score = (df['close'] - df['ma']) / df['std'].replace(0, np.nan)
    return z_score.replace([np.inf, -np.inf], 0)


def calculate_volume_delta(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Delta: Hacim değişimi"""
    vol_delta = df['volume'].pct_change(window)
    return vol_delta.replace([np.inf, -np.inf], 0)


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    """MACD hesapla"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd.replace([np.inf, -np.inf], 0)


def calculate_ema(df: pd.DataFrame, window=20) -> pd.Series:
    """EMA hesapla"""
    return df['close'].ewm(span=window).mean()


def calculate_bollinger_upper(df: pd.DataFrame, window=20) -> pd.Series:
    """Bollinger Upper Band"""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    upper_band = df['ma'] + (2 * df['std'])
    upper_band = upper_band.replace([np.inf, -np.inf], np.nan)
    return upper_band.fillna(df['close'])


def calculate_bollinger_lower(df: pd.DataFrame, window=20) -> pd.Series:
    """Bollinger Lower Band"""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    lower_band = df['ma'] - (2 * df['std'])
    lower_band = lower_band.replace([np.inf, -np.inf], np.nan)
    return lower_band.fillna(df['close'])


def calculate_sma(df: pd.DataFrame, window=20) -> pd.Series:
    """SMA hesapla"""
    return df['close'].rolling(window=window).mean()


def calculate_momentum(df: pd.DataFrame, window=10) -> pd.Series:
    """Momentum hesapla"""
    return df['close'] - df['close'].shift(window)


def calculate_stochastic(df: pd.DataFrame, window=14) -> pd.Series:
    """Stochastic hesapla"""
    df = df.copy()
    df['low_min'] = df['low'].rolling(window=window).min()
    df['high_max'] = df['high'].rolling(window=window).max()
    stochastic = 100 * (df['close'] - df['low_min']) / (df['high_max'] - df['low_min'])
    return stochastic.replace([np.inf, -np.inf], 50)


def calculate_williams_r(df: pd.DataFrame, window=14) -> pd.Series:
    """Williams %R hesapla"""
    df = df.copy()
    df['low_min'] = df['low'].rolling(window=window).min()
    df['high_max'] = df['high'].rolling(window=window).max()
    williams_r = -100 * (df['high_max'] - df['close']) / (df['high_max'] - df['low_min'])
    return williams_r.replace([np.inf, -np.inf], -50)


def live_btc_improved_bot():
    """BTC geliştirilmiş canlı yön tahmin botu"""
    symbol = "BTCUSDT"
    
    client = CryptoExchangeClient()
    
    # Geliştirilmiş fiyat tahmin modellerini yükle
    import joblib
    model_price_4h = joblib.load(f"price_prediction_BTC_4h_improved.pkl")
    model_price_1d = joblib.load(f"price_prediction_BTC_1d_improved.pkl")
    
    # Features
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                'stochastic', 'williams_r']
    
    logger.info("=" * 60)
    logger.info("BTC GELİŞTİRİLMİŞ CANLI YÖN TAHMİN BOTU BAŞLADI")
    logger.info("=" * 60)
    logger.info("4h Walk-Forward: %100.46")
    logger.info("1gün Walk-Forward: %332.79")
    logger.info("4h Stres Test: %58.45")
    logger.info("1gün Stres Test: %211.17")
    logger.info("=" * 60)
    
    last_signal_4h = None
    last_signal_1d = None
    last_price = None
    
    while True:
        try:
            # Veri çek (son 1 ay)
            df = client.fetch_ohlcv(symbol, timeframe="1h", limit=720)
            
            if df is None or df.empty:
                logger.error("Veri çekilemedi")
                time.sleep(300)
                continue
            
            # Inf değerlerini temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < 20:
                logger.error("Yetersiz veri")
                time.sleep(300)
                continue
            
            # Features hesapla
            df['rsi'] = calculate_rsi(df)
            df['atr'] = calculate_atr(df)
            df['z_score'] = calculate_z_score(df)
            df['volume_delta'] = calculate_volume_delta(df)
            df['macd'] = calculate_macd(df)
            df['ema'] = calculate_ema(df)
            df['bollinger_upper'] = calculate_bollinger_upper(df)
            df['bollinger_lower'] = calculate_bollinger_lower(df)
            df['sma'] = calculate_sma(df)
            df['momentum'] = calculate_momentum(df)
            df['stochastic'] = calculate_stochastic(df)
            df['williams_r'] = calculate_williams_r(df)
            
            # Son veriyi al
            last_row = df.iloc[-1]
            current_price = last_row['close']
            
            # Fiyat tahminleri yap
            X = df[features].iloc[-1:]
            
            try:
                pred_price_4h = model_price_4h.predict(X)[0]
                pred_price_1d = model_price_1d.predict(X)[0]
                
                # Yön tahminlerini fiyat tahmininden türet
                signal_4h = 1 if pred_price_4h > current_price else 0
                signal_1d = 1 if pred_price_1d > current_price else 0
                
                # Sinyal değişti mi?
                signal_4h_changed = (last_signal_4h != signal_4h)
                signal_1d_changed = (last_signal_1d != signal_1d)
                price_changed = (last_price != current_price)
                
                # Sinyal veya fiyat değiştiyse bildir
                if signal_4h_changed or signal_1d_changed or price_changed:
                    logger.info("=" * 60)
                    logger.info(f"Zaman: {last_row['timestamp']}")
                    logger.info(f"Fiyat: ${current_price:.2f}")
                    logger.info("=" * 60)
                    
                    if signal_4h == 1:
                        logger.info(f"4h: YÜKSELİŞ bekleniyor (Walk-Forward: %100.46) | Tahmin: ${pred_price_4h:.2f}")
                    else:
                        logger.info(f"4h: DÜŞÜŞ bekleniyor (Walk-Forward: %100.46) | Tahmin: ${pred_price_4h:.2f}")
                    
                    if signal_1d == 1:
                        logger.info(f"1gün: YÜKSELİŞ bekleniyor (Walk-Forward: %332.79) | Tahmin: ${pred_price_1d:.2f}")
                    else:
                        logger.info(f"1gün: DÜŞÜŞ bekleniyor (Walk-Forward: %332.79) | Tahmin: ${pred_price_1d:.2f}")
                    
                    logger.info("=" * 60)
                    
                    last_signal_4h = signal_4h
                    last_signal_1d = signal_1d
                    last_price = current_price
                
            except Exception as e:
                logger.error(f"Tahmin hatası: {e}")
            
            # 5 dakika bekle
            time.sleep(300)
            
        except Exception as e:
            logger.error(f"Hata: {e}")
            time.sleep(300)


if __name__ == "__main__":
    live_btc_improved_bot()
