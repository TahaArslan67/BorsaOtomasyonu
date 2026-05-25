"""
BTC Fiyat Tahmin Modeli Eğitimi
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


def train_btc_price_model():
    """BTC fiyat tahmin modeli eğit"""
    symbol = "BTCUSDT"
    
    client = CryptoExchangeClient()
    
    logger.info(f"BTC fiyat tahmin modeli eğitimi: {symbol}")
    
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
    
    # Hedef değişkenleri oluştur
    df['target_4h'] = df['close'].shift(-4)
    df['target_1d'] = df['close'].shift(-24)
    
    # NaN temizle
    df = df.dropna(subset=['target_4h', 'target_1d'])
    
    logger.info(f"Hedefler oluşturuldu: {len(df)} bar")
    
    # Features
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
    
    # 4h için model eğit
    logger.info("4h fiyat tahmin modeli eğitiliyor...")
    X_4h = df[features]
    y_4h = df['target_4h']
    
    from sklearn.model_selection import train_test_split
    X_train_4h, X_test_4h, y_train_4h, y_test_4h = train_test_split(
        X_4h, y_4h, test_size=0.2, random_state=42
    )
    
    import xgboost as xgb
    model_4h = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        tree_method='hist'
    )
    
    model_4h.fit(X_train_4h, y_train_4h)
    
    # Tahmin yap
    y_pred_4h = model_4h.predict(X_test_4h)
    
    # Performans hesapla
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae_4h = mean_absolute_error(y_test_4h, y_pred_4h)
    mse_4h = mean_squared_error(y_test_4h, y_pred_4h)
    r2_4h = r2_score(y_test_4h, y_pred_4h)
    
    logger.info(f"4h Model Performansı:")
    logger.info(f"  MAE: {mae_4h:.2f}")
    logger.info(f"  MSE: {mse_4h:.2f}")
    logger.info(f"  R²: {r2_4h:.4f}")
    
    # 1gün için model eğit
    logger.info("1gün fiyat tahmin modeli eğitiliyor...")
    X_1d = df[features]
    y_1d = df['target_1d']
    
    X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(
        X_1d, y_1d, test_size=0.2, random_state=42
    )
    
    model_1d = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        tree_method='hist'
    )
    
    model_1d.fit(X_train_1d, y_train_1d)
    
    # Tahmin yap
    y_pred_1d = model_1d.predict(X_test_1d)
    
    # Performans hesapla
    mae_1d = mean_absolute_error(y_test_1d, y_pred_1d)
    mse_1d = mean_squared_error(y_test_1d, y_pred_1d)
    r2_1d = r2_score(y_test_1d, y_pred_1d)
    
    logger.info(f"1gün Model Performansı:")
    logger.info(f"  MAE: {mae_1d:.2f}")
    logger.info(f"  MSE: {mse_1d:.2f}")
    logger.info(f"  R²: {r2_1d:.4f}")
    
    # Modelleri kaydet
    import joblib
    joblib.dump(model_4h, f"price_prediction_BTC_4h.pkl")
    joblib.dump(model_1d, f"price_prediction_BTC_1d.pkl")
    
    logger.info("=" * 60)
    logger.info("BTC FİYAT TAHMİN MODELLERİ KAYDEDİLDİ")
    logger.info("=" * 60)
    logger.info(f"4h Model: price_prediction_BTC_4h.pkl")
    logger.info(f"1gün Model: price_prediction_BTC_1d.pkl")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    train_btc_price_model()
