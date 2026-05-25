import pandas as pd
import numpy as np
from gmstr_system.data_loader import load_and_prepare
from gmstr_system.features import FeatureEngineer, create_targets

df = load_and_prepare()
print(f'Veri: {len(df)} satir')
print(f'Fiyat araligi: {df["Close"].min():.2f} - {df["Close"].max():.2f}')

# Günlük getiriler
ret = df['Close'].pct_change()
print(f'\nGünlük getiri istatistikleri:')
print(ret.describe())
print(f'Pozitif günler: {(ret > 0).sum()} / {len(ret)} = {(ret > 0).mean():.2%}')
print(f'Negatif günler: {(ret < 0).sum()} / {len(ret)} = {(ret < 0).mean():.2%}')
print(f'Sıfır günler: {(ret == 0).sum()} / {len(ret)} = {(ret == 0).mean():.2%}')

# 3 günlük getiriler
ret3 = df['Close'].pct_change(3)
print(f'\n3-günlük getiri istatistikleri:')
print(ret3.describe())
print(f'Pozitif: {(ret3 > 0).sum()} / {len(ret3)} = {(ret3 > 0).mean():.2%}')

# İlk ve son 20 kapanış fiyatı
print(f'\nSon 20 kapanış:')
print(df['Close'].tail(20).values)
