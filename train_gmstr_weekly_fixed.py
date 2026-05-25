"""
GMSTR Haftalık Model Eğitimi - Güncel Verilerle
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import logging
import yfinance as yf
import joblib

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

logger.info("GMSTR günlük veri çekiliyor (10 yıl)...")
ticker = yf.Ticker("GMSTR.IS")
df_daily = ticker.history(period="10y", interval="1d")

# Haftalık resample
df_weekly = df_daily.resample('W-FRI').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna().rename(columns={
    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
}).reset_index()

df_weekly['timestamp'] = df_weekly['Date'].astype('int64') // 10**9
df_weekly = df_weekly[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

logger.info(f"Toplam {len(df_weekly)} haftalık bar ({df_weekly['timestamp'].iloc[0]}-{df_weekly['timestamp'].iloc[-1]})")

# Teknik göstergeler
df = compute_all(df_weekly)
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Feature'lar - hedef 1 hafta sonrası fiyat
features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
df['target'] = df['close'].shift(-1)
df = df.dropna(subset=['target'])

X = df[features]
y = df['target']

# En son veriyi ayır (test için)
X_train, X_test = X[:-20], X[-20:]
y_train, y_test = y[:-20], y[-20:]

logger.info(f"Eğitim: {len(X_train)}, Test: {len(X_test)}")

import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=2.0,
    random_state=42, tree_method='hist'
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Performans
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Test setindeki son 3 haftanın tahminlerini göster
logger.info("=" * 60)
logger.info("TEST SETİ - SON 20 HAFTA TAHMİNLERİ:")
logger.info("=" * 60)
for i in range(min(20, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    err_pct = abs((predicted - actual) / actual) * 100
    direction = "✓" if (predicted > y_test.iloc[max(0,i-1)] and actual > y_test.iloc[max(0,i-1)]) or (predicted <= y_test.iloc[max(0,i-1)] and actual <= y_test.iloc[max(0,i-1)]) else "✗"
    logger.info(f"  Hafta {i+1}: Gerçek={actual:.2f} | Tahmin={predicted:.2f} | Hata=%{err_pct:.1f} | Yön={direction}")

# Şu anki tahmin (en son bar)
current_X = X.iloc[-1:].values.reshape(1, -1)
if hasattr(model, 'feature_names_in_'):
    current_X = X.iloc[-1:]
current_pred = model.predict(current_X)[0]
current_price = df['close'].iloc[-1]
signal = "\U0001f7e2 YÜKSELİŞ" if current_pred > current_price else "\U0001f535 DÜŞÜŞ"
logger.info(f"\nGüncel Fiyat: {current_price:.2f}")
logger.info(f"Haftalık Tahmin: {current_pred:.2f}")
logger.info(f"Sinyal: {signal}")
logger.info(f"Model Performansı - MAE: {mae:.2f} TL, R²: {r2:.4f}")

# Model kaydet
model_path = "price_prediction_GMSTR_1w_updated.pkl"
joblib.dump(model, model_path)
logger.info(f"\nModel kaydedildi: {model_path}")