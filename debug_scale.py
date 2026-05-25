import pandas as pd
import numpy as np

# areaxdatetime
area = pd.read_csv('claude/areaxdatetime.csv')
area['Date'] = pd.to_datetime(area['category'], format='%a %b %d %Y')
area = area.set_index('Date').sort_index()
area['fund_ret'] = pd.to_numeric(area['Net Getiri'], errors='coerce')

# gercek_data -> daily
raw = pd.read_csv('claude/gercek_data.csv', header=None)
header_row = raw.iloc[0].tolist()
data = raw.iloc[3:].copy()
data.columns = header_row
data = data.rename(columns={'Price': 'Date'})
data = data.set_index('Date')
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.index = pd.to_datetime(data.index, errors='coerce', utc=True)
data = data[data.index.notna()]
data = data.sort_index()
daily = data.resample('D').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
daily.index = daily.index.tz_localize(None)

# Ortak tarihler
common = area.join(daily[['Close']], how='inner')
print(f'Ortak gün sayısı: {len(common)}')
print(f'İlk ortak tarih: {common.index[0]}')
print(f'Son ortak tarih: {common.index[-1]}')

# Lineer regresyon: Close = a + b * fund_ret
x = common['fund_ret'].values
y = common['Close'].values
b = np.cov(x, y)[0,1] / np.var(x)
a = np.mean(y) - b * np.mean(x)
print(f'Regresyon: Close = {a:.4f} + {b:.6f} * fund_ret')

# Test: son değer
last_fund = area['fund_ret'].iloc[-1]
predicted = a + b * last_fund
print(f'Son fund_ret: {last_fund} -> Tahmini fiyat: {predicted:.2f}')
print(f'Gerçek son fiyat: {daily["Close"].iloc[-1]:.2f}')
