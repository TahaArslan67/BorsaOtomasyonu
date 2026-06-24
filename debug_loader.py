import pandas as pd
import numpy as np
from pathlib import Path

# areaxdatetime.csv tarih aralığı
df1 = pd.read_csv('claude/areaxdatetime.csv')
df1['Date'] = pd.to_datetime(df1['Unnamed: 0'], errors='coerce')
df1 = df1.dropna(subset=['Date']).set_index('Date').sort_index()
print(f"areaxdatetime: {len(df1)} satır | {df1.index[0].date()} -> {df1.index[-1].date()}")

# gercek_data.csv tarih aralığı
raw = pd.read_csv('claude/gercek_data.csv', header=None)
first_row = raw.iloc[0].astype(str).tolist()
print(f"gercek_data first row: {first_row[:5]}")
if 'Price' in first_row or 'Date' in first_row or 'Datetime' in first_row:
    header_row = raw.iloc[0].tolist()
    data = raw.iloc[3:].copy()
    data.columns = header_row
    date_col = None
    for c in data.columns:
        if str(c).lower() in ['price', 'date', 'datetime']:
            date_col = c
            break
    if date_col:
        data = data.rename(columns={date_col: 'Date'})
        data = data.set_index('Date')
else:
    data = raw.copy()
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data = data.set_index('Date')

for col in ['Close']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
data = data[data.index.notna()]
data = data[['Close']].dropna().sort_index()
daily = data.resample('D').agg({'Close': 'last'}).dropna()
daily.index = daily.index.tz_localize(None)
print(f"gercek_data daily: {len(daily)} satır | {daily.index[0].date()} -> {daily.index[-1].date()}")

# Ortak tarihler
common = df1.join(daily[['Close']].rename(columns={'Close': 'Close_real'}), how='inner')
print(f"Ortak tarih sayısı: {len(common)}")
if len(common) > 0:
    print(f"Ortak aralık: {common.index[0].date()} -> {common.index[-1].date()}")
    print(f"Net Getiri örnek: {common['Net Getiri'].head(3).tolist()}")
    print(f"Close_real örnek: {common['Close_real'].head(3).tolist()}")
