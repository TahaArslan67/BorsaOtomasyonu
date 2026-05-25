"""
GMSTR Haftalık Fiyat Tahmin Modeli Eğitimi
"""
import numpy as np
import pandas as pd
import logging
from exchange_client_bist import BISTExchangeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    return df['tr'].rolling(window=window).mean()


def calculate_z_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    z_score = (df['close'] - df['ma']) / df['std'].replace(0, np.nan)
    return z_score.replace([np.inf, -np.inf], 0)


def calculate_volume_delta(df: pd.DataFrame, window: int = 20) -> pd.Series:
    vol_delta = df['volume'].pct_change(window)
    return vol_delta.replace([np.inf, -np.inf], 0)


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd.replace([np.inf, -np.inf], 0)


def calculate_ema(df: pd.DataFrame, window=20) -> pd.Series:
    return df['close'].ewm(span=window).mean()


def calculate_bollinger_upper(df: pd.DataFrame, window=20) -> pd.Series:
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    upper_band = df['ma'] + (2 * df['std'])
    return upper_band.replace([np.inf, -np.inf], np.nan).fillna(df['close'])


def calculate_bollinger_lower(df: pd.DataFrame, window=20) -> pd.Series:
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    lower_band = df['ma'] - (2 * df['std'])
    return lower_band.replace([np.inf, -np.inf], np.nan).fillna(df['close'])


def calculate_sma(df: pd.DataFrame, window=20) -> pd.Series:
    return df['close'].rolling(window=window).mean()


def calculate_momentum(df: pd.DataFrame, window=10) -> pd.Series:
    return df['close'] - df['close'].shift(window)


def train_gmstr_weekly_model():
    """GMSTR haftalık fiyat tahmin modeli eğit"""
    symbol = "GMSTR"
    client = BISTExchangeClient()

    logger.info(f"{symbol} haftalık fiyat tahmin modeli eğitimi")

    # Haftalık veri çek (mümkün olduğunca fazla - 3 yıl = 156 hafta)
    df = client.fetch_ohlcv(symbol, timeframe="1w", limit=208)

    if df is None or df.empty:
        logger.error(f"Veri çekilemedi: {symbol}")
        return False

    logger.info(f"Veri çekildi: {len(df)} bar")

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

    # Hedef: 1 hafta sonrası fiyat
    df['target_1w'] = df['close'].shift(-1)

    # NaN temizle
    df = df.dropna(subset=['target_1w'])
    logger.info(f"Hedefler oluşturuldu: {len(df)} bar")

    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema',
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']

    # Model eğit
    logger.info("Haftalık fiyat tahmin modeli eğitiliyor...")
    X = df[features]
    y = df['target_1w']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        tree_method='hist'
    )

    model.fit(X_train, y_train)

    # Performans
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Haftalık Model Performansı:")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  MSE: {mse:.2f}")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  Örnek sayısı - Eğitim: {len(X_train)}, Test: {len(X_test)}")

    # Model kaydet
    import joblib
    model_path = f"price_prediction_GMSTR_1w_updated.pkl"
    joblib.dump(model, model_path)

    logger.info("=" * 60)
    logger.info(f"MODEL KAYDEDİLDİ: {model_path}")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    train_gmstr_weekly_model()