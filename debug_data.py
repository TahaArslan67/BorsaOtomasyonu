import pandas as pd

raw = pd.read_csv('claude/gercek_data_5y_1d.csv', header=None)
first_row = raw.iloc[0].astype(str).tolist()
if 'Price' in first_row:
    data = raw.iloc[3:].copy()
    data.columns = first_row
    data = data.rename(columns={'Price': 'Date'})
    data = data.set_index('Date')
else:
    data = raw.copy()
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data = data.set_index('Date')

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
print(f'Toplam satir: {len(data)}')
print(f'Unique Close degerleri: {data["Close"].nunique()}')
print('\nEn cok tekrar eden 10 fiyat:')
print(data['Close'].value_counts().head(10))
print('\nTum fiyat tekrar dagilimi:')
vc = data['Close'].value_counts()
print(f'1 kez: {(vc == 1).sum()}')
print(f'2-4 kez: {((vc >= 2) & (vc <= 4)).sum()}')
print(f'5-9 kez: {((vc >= 5) & (vc <= 9)).sum()}')
print(f'10-19 kez: {((vc >= 10) & (vc <= 19)).sum()}')
print(f'20+ kez: {(vc >= 20).sum()}')

# Ardışık tekrar kontrolü
dup = data['Close'].eq(data['Close'].shift())
runs = dup.groupby((~dup).cumsum()).transform('sum')
print(f'\nArdisik tekrar dagilimi:')
print(f'1-4 gun: {((runs >= 1) & (runs <= 4)).sum()}')
print(f'5-9 gun: {((runs >= 5) & (runs <= 9)).sum()}')
print(f'10-19 gun: {((runs >= 10) & (runs <= 19)).sum()}')
print(f'20+ gun: {(runs >= 20).sum()}')
