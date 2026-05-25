"""
GMSTR Gelişmiş Haftalık Model Eğitimi
Günlük veriden haftalık bar oluşturarak çok daha fazla veri noktası elde eder (256+ hafta)
"""
import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50)


def calculate_atr(df, window=14):
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    return df['tr'].rolling(window=window).mean()


def calculate_z_score(df, window=20):
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    z_score = (df['close'] - df['ma']) / df['std'].replace(0, np.nan)
    return z_score.replace([np.inf, -np.inf], 0)


def calculate_volume_delta(df, window=20):
    vol_delta = df['volume'].pct_change(window)
    return vol_delta.replace([np.inf, -np.inf], 0)


def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['close'].ewm(span=slow, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    return macd.replace([np.inf, -np.inf], 0)


def calculate_ema(df, window=20):
    return df['close'].ewm(span=window, min_periods=window).mean()


def calculate_bollinger_upper(df, window=20):
    ma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = ma + (2 * std)
    return upper_band.replace([np.inf, -np.inf], np.nan).fillna(df['close'])


def calculate_bollinger_lower(df, window=20):
    ma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    lower_band = ma - (2 * std)
    return lower_band.replace([np.inf, -np.inf], np.nan).fillna(df['close'])


def calculate_sma(df, window=20):
    return df['close'].rolling(window=window).mean()


def calculate_momentum(df, window=10):
    return df['close'] - df['close'].shift(window)


def compute_technicals(df):
    """Tüm teknik göstergeleri hesapla"""
    df = df.copy()
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
    df = df.ffill().bfill()
    return df


def train_weekly_model():
    symbol = "GMSTR"
    
    import yfinance as yf
    
    logger.info(f"{symbol} - Günlük veri çekiliyor...")
    ticker = yf.Ticker(f"{symbol}.IS")
    
    # 10 yıllık günlük veri çek
    df_daily = ticker.history(period="10y", interval="1d")
    
    if df_daily is None or df_daily.empty:
        logger.error("Günlük veri çekilemedi!")
        return False
    
    logger.info(f"Günlük veri: {len(df_daily)} bar")
    
    # Günlük veriyi haftalık OHLCV'ye resample et
    df_weekly = df_daily.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Sütun adlarını düzelt
    df_weekly = df_weekly.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }).reset_index()
    
    df_weekly['timestamp'] = df_weekly['Date'].astype('int64') // 10**9
    df_weekly = df_weekly[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    logger.info(f"Haftalık veri: {len(df_weekly)} bar (ilk: {pd.to_datetime(df_weekly['timestamp'].iloc[0], unit='s').date()}, son: {pd.to_datetime(df_weekly['timestamp'].iloc[-1], unit='s').date()})")
    
    # Teknik göstergeleri hesapla
    df = compute_technicals(df_weekly)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Hedef: 1 hafta sonrası fiyat
    df['target'] = df['close'].shift(-1)
    df = df.dropna(subset=['target'])
    
    logger.info(f"Eğitim için hazır: {len(df)} bar")
    
    # Feature'lar
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema',
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
    
    X = df[features]
    y = df['target']
    
    # Train/Test split (zaman sıralı - en son %20 test)
    test_size = int(len(df) * 0.2)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    logger.info(f"Eğitim: {len(X_train)}, Test: {len(X_test)} bar")
    
    # Model eğit - hyperparameter tuning
    import xgboost as xgb
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Test performansı
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info("=" * 60)
    logger.info(f"MODEL PERFORMANSI")
    logger.info("=" * 60)
    logger.info(f"  MAE:  {mae:.2f} TL")
    logger.info(f"  RMSE: {rmse:.2f} TL")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  Ort. Fiyat: {y_test.mean():.2f} TL")
    logger.info(f"  MAE/Fiyat: %{(mae / y_test.mean()) * 100:.1f}")
    
    # Direction accuracy (yön doğruluğu)
    current_prices = X_test['sma'].values  # proxy olarak
    actual_direction = (y_test.values > 0).astype(int)  # fiyat > 0 her zaman true, daha iyi karşılaştırma:
    
    # Yön doğruluğu: tahmin edilen fiyat değişimi ile gerçek fiyat değişimi karşılaştırması
    actual_changes = np.diff(np.concatenate([[X_train['sma'].iloc[-1]], y_test.values]))
    pred_changes = np.diff(np.concatenate([[X_train['sma'].iloc[-1]], y_pred]))
    
    actual_direction = actual_changes > 0
    pred_direction = pred_changes > 0
    
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    logger.info(f"  Yön Doğruluğu: %{direction_accuracy:.1f}")
    
    # Alım/Satım sinyali doğruluğu (daha anlamlı)
    correct_buy = np.sum((pred_direction == True) & (actual_direction == True))
    correct_sell = np.sum((pred_direction == False) & (actual_direction == False))
    total_signals = len(actual_direction)
    signal_accuracy = (correct_buy + correct_sell) / total_signals * 100
    logger.info(f"  AL/SAT Doğruluğu: %{signal_accuracy:.1f}")
    logger.info(f"  Doğru AL: {correct_buy}, Doğru SAT: {correct_sell}, Toplam: {total_signals}")
    
    # Model kaydet
    import joblib
    model_path = "price_prediction_GMSTR_1w_updated.pkl"
    joblib.dump(model, model_path)
    
    logger.info("=" * 60)
    logger.info(f"MODEL KAYDEDİLDİ: {model_path}")
    logger.info("=" * 60)
    
    # prediction vs actual plot için örnek
    results_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'actual_dir': actual_direction,
        'pred_dir': pred_direction
    })
    results_df.to_csv("gmstr_weekly_test_results.csv", index=False)
    logger.info("Test sonuçları kaydedildi: gmstr_weekly_test_results.csv")
    
    return True


if __name__ == "__main__":
    train_weekly_model()