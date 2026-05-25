import pandas as pd
import numpy as np
from gmstr_system.data_loader import load_and_prepare
from gmstr_system.features import FeatureEngineer, create_targets

df = load_and_prepare()
print(f'Ham veri: {len(df)}')
print(f'Volume 0 sayisi: {(df["Volume"]==0).sum()}')

eng = FeatureEngineer()
df2 = eng.transform(df)
print(f'Feature sonrasi: {len(df2)}')
print(f'NaN sayisi (toplam): {df2.isna().sum().sum()}')

nan_counts = df2.isna().sum().sort_values(ascending=False)
print('En cok NaN olan kolonlar:')
print(nan_counts.head(20))

# Volume 0 olan günlerin indikatörlerinde sorun var mı?
df3 = create_targets(df2, {'1d':1, '3d':3, '5d':5, '10d':10}, 0.0005)
feat_cols = eng.get_feature_columns(df3)
clean = df3.dropna(subset=feat_cols + ['target_1d'])
print(f'Temiz veri (1d): {len(clean)}')
print(f'Toplam feature: {len(feat_cols)}')

# Hangi satırlar NaN?
if len(clean) == 0:
    sample = df3[feat_cols + ['target_1d']].isna().sum(axis=1)
    print(f'NaN içeren satır sayısı (herhangi bir kolonda): {(sample > 0).sum()}')
    print(f'Hepsi NaN olan satır: {(sample == len(feat_cols)+1).sum()}')
