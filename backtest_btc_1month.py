"""
BTC Son 1 Ay Backtest
"""

import numpy as np
import pandas as pd
import logging
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


def backtest_btc_1month():
    """BTC son 1 ay backtest"""
    symbol = "BTCUSDT"
    
    client = CryptoExchangeClient()
    
    logger.info(f"BTC son 1 ay backtest: {symbol}")
    
    # Veri çek (son 1 ay)
    df = client.fetch_ohlcv(symbol, timeframe="1h", limit=720)
    
    if df is None or df.empty:
        logger.error(f"Veri çekilemedi: {symbol}")
        return False
    
    logger.info(f"Veri çekildi: {len(df)} bar")
    
    # Inf değerlerini temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    logger.info(f"Veri temizlendi: {len(df)} bar")
    
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
    
    # Yön tahmin modellerini yükle
    import joblib
    model_4h = joblib.load(f"direction_prediction_BTC_4h.pkl")
    model_1d = joblib.load(f"direction_prediction_BTC_1d.pkl")
    
    # Features
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
    
    # Son 1 ay için test
    test_data = df.copy()
    
    logger.info("=" * 60)
    logger.info("BTC SON 1 AY YÖN TAHMİN DOĞRULUĞU TESTİ")
    logger.info("=" * 60)
    
    # 4h test
    correct_4h = 0
    total_4h = 0
    
    for i in range(len(test_data) - 4):
        X = test_data[features].iloc[i:i+1]
        current_price = test_data['close'].iloc[i]
        
        if i + 4 < len(test_data):
            pred_4h = model_4h.predict(X)[0]
            actual_4h = test_data['close'].iloc[i+4]
            
            pred_direction = 1 if pred_4h == 1 else -1
            actual_direction = 1 if actual_4h > current_price else -1
            
            if pred_direction == actual_direction:
                correct_4h += 1
            
            total_4h += 1
    
    accuracy_4h = (correct_4h / total_4h * 100) if total_4h > 0 else 0
    
    # 1gün test
    correct_1d = 0
    total_1d = 0
    
    for i in range(len(test_data) - 24):
        X = test_data[features].iloc[i:i+1]
        current_price = test_data['close'].iloc[i]
        
        if i + 24 < len(test_data):
            pred_1d = model_1d.predict(X)[0]
            actual_1d = test_data['close'].iloc[i+24]
            
            pred_direction = 1 if pred_1d == 1 else -1
            actual_direction = 1 if actual_1d > current_price else -1
            
            if pred_direction == actual_direction:
                correct_1d += 1
            
            total_1d += 1
    
    accuracy_1d = (correct_1d / total_1d * 100) if total_1d > 0 else 0
    
    logger.info(f"4h Tahmin Doğruluğu: %{accuracy_4h:.2f} ({correct_4h}/{total_4h})")
    logger.info(f"1gün Tahmin Doğruluğu: %{accuracy_1d:.2f} ({correct_1d}/{total_1d})")
    logger.info("=" * 60)
    
    if accuracy_4h >= 65:
        logger.info("4h modeli doğruluğu %65+ hedefini aşıyor")
    else:
        logger.info(f"4h modeli doğruluğu %65+ hedefine ulaşmak için %{(65.0 - accuracy_4h):.2f} daha gerekiyor")
    
    if accuracy_1d >= 65:
        logger.info("1gün modeli doğruluğu %65+ hedefini aşıyor")
    else:
        logger.info(f"1gün modeli doğruluğu %65+ hedefine ulaşmak için %{(65.0 - accuracy_1d):.2f} daha gerekiyor")
    
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    backtest_btc_1month()
