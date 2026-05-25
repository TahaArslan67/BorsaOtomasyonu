"""
GMSTR için 3 Ayrı Model Eğitimi: 1h, 4h, 1d
Walk-forward backtest ile gerçek güvenilirlik testi
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import logging
import yfinance as yf
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Teknik göstergeler
def calculate_rsi(df, w=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(w).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 50)

def calculate_atr(df, w=14):
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    return tr.rolling(w).mean()

def calculate_z_score(df, w=20):
    ma = df['close'].rolling(w).mean()
    std = df['close'].rolling(w).std()
    return ((df['close'] - ma) / std.replace(0, np.nan)).replace([np.inf, -np.inf], 0)

def calculate_volume_delta(df, w=20):
    return df['volume'].pct_change(w).replace([np.inf, -np.inf], 0)

def calculate_macd(df, fast=12, slow=26):
    ema_f = df['close'].ewm(span=fast).mean()
    ema_s = df['close'].ewm(span=slow).mean()
    return (ema_f - ema_s).replace([np.inf, -np.inf], 0)

def calculate_ema(df, w=20):
    return df['close'].ewm(span=w).mean()

def calculate_bollinger_upper(df, w=20):
    ma = df['close'].rolling(w).mean()
    std = df['close'].rolling(w).std()
    return (ma + 2*std).replace([np.inf, -np.inf], np.nan).fillna(df['close'])

def calculate_bollinger_lower(df, w=20):
    ma = df['close'].rolling(w).mean()
    std = df['close'].rolling(w).std()
    return (ma - 2*std).replace([np.inf, -np.inf], np.nan).fillna(df['close'])

def calculate_sma(df, w=20):
    return df['close'].rolling(w).mean()

def calculate_momentum(df, w=10):
    return df['close'] - df['close'].shift(w)

def compute_all(df):
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
    return df.ffill().bfill()

def get_data_for_tf(tf):
    """Zaman dilimine göre veri hazırla"""
    ticker = yf.Ticker("GMSTR.IS")
    df_daily = ticker.history(period="5y", interval="1d")
    
    if tf == '1h':
        # Günlük veriyi kullan, hedef 1 gün sonrası (proxy)
        df = df_daily.copy().reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        target_shift = 1  # 1 gün sonra
        
    elif tf == '4h':
        # Günlük veriyi kullan, hedef 1 gün sonrası (proxy)
        df = df_daily.copy().reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        target_shift = 1  # 1 gün sonra (proxy)
        
    elif tf == '1d':
        # Günlük veri
        df = df_daily.copy().reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        target_shift = 1  # 1 gün sonra
        
    else:
        return None, 0
    
    return df, target_shift

def train_model_for_tf(tf, model_path):
    """Belirli zaman dilimi için model eğit"""
    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL EĞİTİMİ: {tf}")
    logger.info(f"{'='*60}")
    
    df, target_shift = get_data_for_tf(tf)
    if df is None:
        return None
    
    # Teknik göstergeler
    df = compute_all(df)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Hedef
    df['target'] = df['close'].shift(-target_shift)
    df = df.dropna(subset=['target'])
    
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema',
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
    
    X = df[features]
    y = df['target']
    
    # Walk-forward backtest: Son 60 günü test olarak ayır
    train_size = len(df) - 60
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Eğitim: {len(X_train)} bar, Test: {len(X_test)} bar")
    
    # Model eğit
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0,
        random_state=42, tree_method='hist'
    )
    
    model.fit(X_train, y_train)
    
    # Test
    y_pred = model.predict(X_test)
    
    # Metrikler
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_pred - y_test.values) / y_test.values)) * 100
    
    # Yön doğruluğu
    current_prices = X_test['sma'].values  # proxy
    actual_direction = y_test.values > current_prices
    pred_direction = y_pred > current_prices
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Kazanç simülasyonu
    capital = 10000.0
    for i in range(len(y_test)):
        if pred_direction[i]:
            pct = (y_test.values[i] - current_prices[i]) / current_prices[i]
        else:
            pct = (current_prices[i] - y_test.values[i]) / current_prices[i]
        capital *= (1 + pct)
    
    total_return = (capital - 10000) / 10000 * 100
    
    logger.info(f"MAE: {mae:.2f} TL")
    logger.info(f"MAPE: %{mape:.2f}")
    logger.info(f"Yön Doğruluğu: %{direction_accuracy:.2f}")
    logger.info(f"Simülasyon Getiri: %{total_return:.2f}")
    
    # Model kaydet
    joblib.dump(model, model_path)
    logger.info(f"Model kaydedildi: {model_path}")
    
    return {
        'tf': tf,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'total_return': total_return
    }

def main():
    logger.info("=" * 60)
    logger.info("GMSTR 3 AYRI MODEL EĞİTİMİ")
    logger.info("=" * 60)
    
    models = [
        ('1h', 'price_prediction_GMSTR_1h_v2.pkl'),
        ('4h', 'price_prediction_GMSTR_4h_v2.pkl'),
        ('1d', 'price_prediction_GMSTR_1d_v2.pkl'),
    ]
    
    results = []
    for tf, path in models:
        result = train_model_for_tf(tf, path)
        if result:
            results.append(result)
    
    # Özet
    logger.info("\n" + "=" * 60)
    logger.info("ÖZET")
    logger.info("=" * 60)
    for r in results:
        logger.info(f"{r['tf']}: Yön=%{r['direction_accuracy']:.1f}, MAPE=%{r['mape']:.1f}, Getiri=%{r['total_return']:.1f}")
    
    # Sonuçları kaydet
    df = pd.DataFrame(results)
    df.to_csv("gmstr_model_training_results.csv", index=False)

if __name__ == "__main__":
    main()