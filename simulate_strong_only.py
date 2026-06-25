"""
Sadece guclu sinyallerle simulasyon (40,000 TL)
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from gmstr_prediction_system import GMSTRPredictionSystem
import warnings
warnings.filterwarnings('ignore')

ps = GMSTRPredictionSystem()

print("=" * 70)
print("GUCLU SINYAL SIMULASYONU - 40,000 TL")
print("Sadece |proba-0.5| > 0.25 olanlar")
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

# Walk-forward: son 20%
split = int(len(X_all) * 0.8)
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
trades = []

for i in range(len(X_test)):
    price = gmstr['Close'].iloc[-len(X_test)+i]
    date = gmstr.index[-len(X_test)+i]
    actual = y_test[i]
    
    proba = model.predict_proba(X_test_p[i].reshape(1, -1))[0][1]
    pred = 1 if proba > 0.5 else 0
    
    # Sadece guclu sinyaller (>0.75 veya <0.25)
    is_strong = (proba > 0.75) or (proba < 0.25)
    
    if not is_strong:
        continue
    
    if pred == 1 and position is None:
        # AL
        shares = int(cash / price)
        if shares > 0:
            cash -= shares * price
            position = "LONG"
            entry_price = price
            correct = (actual == 1)
            trades.append({'type': 'AL', 'date': str(date)[:10], 'price': price, 'correct': correct, 'conf': proba})
    
    elif pred == 0 and position == "LONG":
        # SAT
        revenue = shares * price
        cash += revenue
        correct = (actual == 0)
        profit = revenue - (shares * entry_price)
        trades.append({'type': 'SAT', 'date': str(date)[:10], 'price': price, 'correct': correct, 'conf': proba, 'profit': profit})
        shares = 0
        position = None

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
print(f"\nIslem Sayisi:  {len(trades)//2}")
print(f"Dogru:         {sum(1 for t in trades if t['correct'])}")
print(f"Yanlis:        {sum(1 for t in trades if not t['correct'])}")

if trades:
    profits = [t.get('profit', 0) for t in trades if 'profit' in t]
    if profits:
        print(f"Toplam K/Z:    {sum(profits):+,.0f} TL")
        print(f"Ortalama K/Z:  {np.mean(profits):+,.0f} TL")
    
    print(f"\n{'='*70}")
    print("ISLEMLER")
    print('='*70)
    for t in trades:
        if t['type'] == 'AL':
            print(f"AL  {t['date']}  Fiyat: {t['price']:.2f}  Guven: {t['conf']*100:.1f}%")
        else:
            print(f"SAT {t['date']}  Fiyat: {t['price']:.2f}  Guven: {t['conf']*100:.1f}%  K/Z: {t.get('profit', 0):+,.0f} TL")
