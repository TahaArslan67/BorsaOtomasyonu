"""
Sadece ADX ile GMSTR backtest
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# GMSTR verisi
gmstr = yf.Ticker("GMSTR.IS")
df = gmstr.history(period="2y", interval="4h")
print(f"Veri: {len(df)} bar | {df.index[0]} -> {df.index[-1]}")

# ADX hesapla
def calculate_adx(df, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    
    return adx, plus_di, minus_di

adx, plus_di, minus_di = calculate_adx(df)
df['adx'] = adx
df['plus_di'] = plus_di
df['minus_di'] = minus_di

# NaN temizligi
df = df.dropna()
print(f"ADX sonrasi: {len(df)} bar")

# Stratejiler
def backtest(df, threshold=25, use_direction=True):
    """ADX backtest"""
    cash = 40000
    shares = 0
    position = None  # None, 'LONG', 'SHORT'
    values = []
    trades = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['Close']
        adx_val = row['adx']
        plus = row['plus_di']
        minus = row['minus_di']
        
        # Trend mi?
        is_trend = adx_val >= threshold
        
        # Yon
        if plus > minus:
            direction = 'UP'
        else:
            direction = 'DOWN'
        
        # Sinyal
        if is_trend:
            if direction == 'UP' and position != 'LONG':
                # AL
                if position == 'SHORT':
                    # Once kapat
                    cash += shares * price
                    shares = 0
                    position = None
                    trades.append(('SAT_KAPAT', df.index[i], price))
                if position is None:
                    shares = int(cash / price)
                    cash -= shares * price
                    position = 'LONG'
                    trades.append(('AL', df.index[i], price))
            
            elif direction == 'DOWN' and position != 'SHORT':
                # SAT (short)
                if position == 'LONG':
                    cash += shares * price
                    shares = 0
                    position = None
                    trades.append(('SAT', df.index[i], price))
                # Short pozisyon (simule: nakitte kal)
                if position is None:
                    position = 'SHORT'
                    trades.append(('SHORT', df.index[i], price))
        
        # Portfoy degeri
        if position == 'LONG':
            val = cash + shares * price
        elif position == 'SHORT':
            val = cash  # Short simulasyonu: nakitte kal
        else:
            val = cash
        
        values.append({'time': df.index[i], 'value': val, 'price': price, 'adx': adx_val, 'dir': direction, 'pos': position})
    
    final_price = df['Close'].iloc[-1]
    if position == 'LONG':
        final_val = cash + shares * final_price
    else:
        final_val = cash
    
    return final_val, trades, values

# Farkli threshold'larla test
for thresh in [20, 25, 30]:
    print(f"\n{'='*60}")
    print(f"ADX >= {thresh} (Trend)")
    print('='*60)
    
    val, trades, vals = backtest(df, threshold=thresh)
    
    ret = (val - 40000) / 40000 * 100
    
    # Buy&Hold
    buyhold = int(40000 / df['Close'].iloc[0]) * df['Close'].iloc[-1]
    bh_ret = (buyhold - 40000) / 40000 * 100
    
    # Istatistikler
    df_vals = pd.DataFrame(vals)
    if len(df_vals) > 0:
        max_dd = ((df_vals['value'].cummax() - df_vals['value']) / df_vals['value'].cummax()).max() * 100
    else:
        max_dd = 0
    
    print(f"Islem sayisi: {len(trades)}")
    print(f"Getiri: %{ret:.2f}")
    print(f"Buy&Hold: %{bh_ret:.2f}")
    print(f"Fark: {ret - bh_ret:+.2f} puan")
    print(f"Max Drawdown: %{max_dd:.2f}")
    
    if trades:
        print("\nSon 5 islem:")
        for t in trades[-5:]:
            print(f"  {t[0]} | {t[1].strftime('%Y-%m-%d %H:%M')} | {t[2]:.2f}")

# ADX degeri dagilimi
print(f"\n{'='*60}")
print("ADX Istatistikleri (tum veri)")
print('='*60)
print(f"Ortalama ADX: {df['adx'].mean():.1f}")
print(f"Median ADX: {df['adx'].median():.1f}")
print(f"ADX >= 25: %{(df['adx'] >= 25).mean()*100:.1f} (trend gunu)")
print(f"ADX >= 20: %{(df['adx'] >= 20).mean()*100:.1f}")
print(f"Max ADX: {df['adx'].max():.1f}")
print(f"Min ADX: {df['adx'].min():.1f}")

# Gunluk ADX dagilimi (son 30 gun)
print(f"\nSon 30 gun ADX:")
for i in range(min(30, len(df))):
    row = df.iloc[-30+i]
    print(f"  {df.index[-30+i].strftime('%m-%d %H:%M')} | ADX={row['adx']:5.1f} | +DI={row['plus_di']:5.1f} | -DI={row['minus_di']:5.1f} | Fiyat={row['Close']:7.2f}")
