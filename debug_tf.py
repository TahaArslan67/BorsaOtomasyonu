import warnings
warnings.filterwarnings('ignore')
from exchange_client_bist import BISTExchangeClient
import joblib
import pandas as pd
import numpy as np
from crypto_monitor_v2 import calculate_rsi, calculate_atr, calculate_z_score, calculate_volume_delta, calculate_macd, calculate_ema, calculate_bollinger_upper, calculate_bollinger_lower, calculate_sma, calculate_momentum

model = joblib.load('price_prediction_GMSTR_1d_updated.pkl')
features = list(model.feature_names_in_)
print("Features:", features)

bist = BISTExchangeClient()

# 1d verisi
df_1d = bist.fetch_ohlcv('GMSTR', timeframe='1d', limit=365)
df_1d = df_1d.replace([np.inf, -np.inf], np.nan).dropna()
df_1d['rsi'] = calculate_rsi(df_1d)
df_1d['atr'] = calculate_atr(df_1d)
df_1d['z_score'] = calculate_z_score(df_1d)
df_1d['volume_delta'] = calculate_volume_delta(df_1d)
df_1d['macd'] = calculate_macd(df_1d)
df_1d['ema'] = calculate_ema(df_1d)
df_1d['bollinger_upper'] = calculate_bollinger_upper(df_1d)
df_1d['bollinger_lower'] = calculate_bollinger_lower(df_1d)
df_1d['sma'] = calculate_sma(df_1d)
df_1d['momentum'] = calculate_momentum(df_1d)
df_1d = df_1d.ffill().bfill()

X_1d = df_1d[features].iloc[-1:]
pred_1d = model.predict(X_1d)[0]
close_1d = df_1d['close'].iloc[-1]
print(f"1d: close={close_1d:.2f} pred={pred_1d:.2f}")

# 1w verisi
df_1w = bist.fetch_ohlcv('GMSTR', timeframe='1w', limit=156)
df_1w = df_1w.replace([np.inf, -np.inf], np.nan).dropna()
df_1w['rsi'] = calculate_rsi(df_1w)
df_1w['atr'] = calculate_atr(df_1w)
df_1w['z_score'] = calculate_z_score(df_1w)
df_1w['volume_delta'] = calculate_volume_delta(df_1w)
df_1w['macd'] = calculate_macd(df_1w)
df_1w['ema'] = calculate_ema(df_1w)
df_1w['bollinger_upper'] = calculate_bollinger_upper(df_1w)
df_1w['bollinger_lower'] = calculate_bollinger_lower(df_1w)
df_1w['sma'] = calculate_sma(df_1w)
df_1w['momentum'] = calculate_momentum(df_1w)
df_1w = df_1w.ffill().bfill()

X_1w = df_1w[features].iloc[-1:]
pred_1w = model.predict(X_1w)[0]
close_1w = df_1w['close'].iloc[-1]
print(f"1w: close={close_1w:.2f} pred={pred_1w:.2f}")

# Feature değerlerini karşılaştır
print("\nFeature comparison (last row):")
for f in features:
    print(f"  {f}: 1d={X_1d[f].values[0]:.4f} | 1w={X_1w[f].values[0]:.4f}")