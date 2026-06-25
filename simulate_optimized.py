"""
Optimize edilmis simulasyon:
- Guclu sinyaller (>0.70 veya <0.30)
- Stop-loss: %3
- Take-profit: %5
- Max drawdown koruma
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from gmstr_prediction_system import GMSTRPredictionSystem
import warnings
warnings.filterwarnings('ignore')

ps = GMSTRPredictionSystem()

print("=" * 70)
print("OPTIMIZE SIMULASYON - 40,000 TL")
print("Guclu sinyal + Stop-loss %3 + Take-profit %5")
print("=" * 70)

# Veri cek
gmstr = ps.fetch_gmstr_data(period="2y")
market = ps.fetch_market_data()

X_all = ps.create_features(gmstr, market)
y_all = ps.create_labels(gmstr)

min_len = min(len(X_all), len(y_all))
X_all = X_all[:min_len]
y_all = y_all[:min_len]
mask = y_all != 2
X_all = X_all[mask]
y_all = y_all[mask]

# Son 6 ay test
split = int(len(X_all) * 0.75)
X_train = X_all[:split]
y_train = y_all[:split]
X_test = X_all[split:]
y_test = y_all[split:]

# PCA
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

pca = PCA(n_components=20, random_state=42)
X_train_p = pca.fit_transform(X_train_s)
X_test_p = pca.transform(X_test_s)

# LGBM
model = lgb.LGBMClassifier(
    n_estimators=50, max_depth=2, learning_rate=0.05,
    subsample=0.6, colsample_bytree=0.6,
    is_unbalance=True,
    reg_alpha=2.0, reg_lambda=5.0,
    min_child_samples=100,
    random_state=42, n_jobs=-1, verbosity=-1
)
model.fit(X_train_p, y_train)

# Simulasyon
cash = 40000
shares = 0
position = None
entry_price = 0
peak_value = 40000

STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.05
MAX_DD_PCT = 0.15

trades = []
portfolio = []

for i in range(len(X_test)):
    price = gmstr['Close'].iloc[-len(X_test)+i]
    date = gmstr.index[-len(X_test)+i]
    
    proba = model.predict_proba(X_test_p[i].reshape(1, -1))[0][1]
    pred = 1 if proba > 0.5 else 0
    
    # Guclu sinyal kontrolu
    is_strong = (proba > 0.70) or (proba < 0.30)
    
    port_val = cash + (shares * price)
    portfolio.append({'date': str(date)[:10], 'value': port_val, 'price': price})
    peak_value = max(peak_value, port_val)
    
    # Max drawdown kontrolu - tumunu sat
    dd = (peak_value - port_val) / peak_value
    if dd > MAX_DD_PCT and position == "LONG":
        cash += shares * price
        trades.append({'type': 'SAT_DD', 'date': str(date)[:10], 'price': price, 'profit': (price - entry_price) * shares})
        shares = 0
        position = None
        continue
    
    if position == "LONG":
        # Stop-loss veya take-profit kontrolu
        change = (price - entry_price) / entry_price
        if change <= -STOP_LOSS_PCT:
            cash += shares * price
            trades.append({'type': 'SAT_SL', 'date': str(date)[:10], 'price': price, 'profit': (price - entry_price) * shares})
            shares = 0
            position = None
        elif change >= TAKE_PROFIT_PCT:
            cash += shares * price
            trades.append({'type': 'SAT_TP', 'date': str(date)[:10], 'price': price, 'profit': (price - entry_price) * shares})
            shares = 0
            position = None
    
    if not is_strong:
        continue
    
    if pred == 1 and position is None:
        shares = int(cash / price)
        if shares > 0:
            cash -= shares * price
            position = "LONG"
            entry_price = price
            trades.append({'type': 'AL', 'date': str(date)[:10], 'price': price, 'conf': proba})

# Son deger
final_price = gmstr['Close'].iloc[-1]
final_val = cash + (shares * final_price)

# Buy&Hold
bh_shares = int(40000 / gmstr['Close'].iloc[-len(X_test)])
bh_val = bh_shares * final_price

# Sonuclar
print(f"\n{'='*70}")
print("SONUCLAR")
print('='*70)
print(f"Baslangic:     40,000 TL")
print(f"Bitis:         {final_val:,.0f} TL")
print(f"Getiri:        %{((final_val-40000)/40000*100):+.2f}")
print(f"Buy&Hold:      {bh_val:,.0f} TL (%{((bh_val-40000)/40000*100):+.2f})")
print(f"Fark:          {((final_val-40000)/40000*100) - ((bh_val-40000)/40000*100):+.2f} puan")

# Islem analizi
buy_count = sum(1 for t in trades if t['type'] == 'AL')
profits = [t.get('profit', 0) for t in trades if 'profit' in t]
wins = sum(1 for p in profits if p > 0)
losses = sum(1 for p in profits if p <= 0)

print(f"\nIslem Sayisi:  {buy_count}")
print(f"Kazanan:       {wins}")
print(f"Kaybeden:      {losses}")
if profits:
    print(f"Toplam K/Z:    {sum(profits):+,.0f} TL")
    print(f"Ortalama K/Z:  {np.mean(profits):+,.0f} TL")
    print(f"Buyuk Kazanc:  {max(profits):+,.0f} TL")
    print(f"Buyuk Kayip:   {min(profits):+,.0f} TL")

# Islem detaylari
print(f"\n{'='*70}")
print("ISLEMLER")
print('='*70)
for t in trades:
    if t['type'] == 'AL':
        print(f"AL   {t['date']}  Fiyat: {t['price']:>8.2f}  Guven: {t['conf']*100:>5.1f}%")
    else:
        pnl = t.get('profit', 0)
        print(f"{t['type']} {t['date']}  Fiyat: {t['price']:>8.2f}  K/Z: {pnl:+10,.0f} TL")

# Aylik ozet
print(f"\n{'='*70}")
print("AYLIK DEGER")
print('='*70)
import pandas as pd
pdf = pd.DataFrame(portfolio)
pdf['date'] = pd.to_datetime(pdf['date'])
monthly = pdf.groupby(pdf['date'].dt.to_period('M')).last()
for period, row in monthly.iterrows():
    print(f"{str(period):<10} | Portfoy: {row['value']:>10,.0f} TL | Fiyat: {row['price']:>8.2f}")
