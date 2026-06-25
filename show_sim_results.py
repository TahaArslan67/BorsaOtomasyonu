import sys
sys.path.insert(0, 'd:\\otonomBorsa')

# Simulate_1year_walkforward modulunu import et ve calistir
# Ama sonuclari burada gosterelim

from gmstr_prediction_system import GMSTRPredictionSystem
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

ps = GMSTRPredictionSystem()

print("=" * 70)
print("1 YILLIK WALK-FORWARD SIMULASYON - SONUCLAR")
print("Baslangic: 40,000 TL")
print("=" * 70)

# Veri cek
gmstr = ps.fetch_gmstr_data(period="2y")
market = ps.fetch_market_data()

X_all = ps.create_features(gmstr, market)
y_all = ps.create_labels(gmstr)

min_len = min(len(X_all), len(y_all))
X_all = X_all[:min_len]
y_all = y_all[:min_len]

dates = gmstr.index[-min_len:]

window_train = 2000
window_test = 200

cash = 40000
shares = 0
position = None
trades = []
portfolio_history = []

total_correct = 0
total_trades = 0

n_samples = len(X_all)
step = 200

windows = []
start = window_train
while start + window_test <= n_samples:
    windows.append((start - window_train, start, start + window_test))
    start += step

print(f"Toplam {len(windows)} donem test edildi")

for idx, (train_start, train_end, test_end) in enumerate(windows):
    X_train = X_all[train_start:train_end]
    y_train = y_all[train_start:train_end]
    X_test = X_all[train_end:test_end]
    y_test = y_all[train_end:test_end]
    test_dates = dates[train_end:test_end]
    test_prices = gmstr['Close'].iloc[train_end:test_end].values

    mask = y_train != 2
    X_train, y_train = X_train[mask], y_train[mask]

    if len(X_train) < 100 or len(X_test) == 0:
        continue

    rf_sel = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train, y_train)
    top_idx = np.argsort(rf_sel.feature_importances_)[-30:]
    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    rf = RandomForestClassifier(n_estimators=80, max_depth=3, min_samples_split=100,
                                min_samples_leaf=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_sel, y_train)

    for i in range(len(X_test)):
        current_price = float(test_prices[i])
        current_date = test_dates[i]
        actual_direction = y_test[i]

        features = X_test_sel[i].reshape(1, -1)
        proba = rf.predict_proba(features)[0][1]
        prediction = 1 if proba > 0.5 else 0

        if abs(proba - 0.5) > 0.15:
            signal = "AL" if prediction == 1 else "SAT"
        else:
            signal = "BEKLE"

        if signal == "AL" and position is None:
            shares = int(cash / current_price)
            if shares > 0:
                cost = shares * current_price
                cash -= cost
                position = "LONG"
                entry_price = current_price

                correct = (actual_direction == 1)
                total_correct += correct
                total_trades += 1

                trades.append({
                    'type': 'AL', 'date': str(current_date), 'price': current_price,
                    'shares': shares, 'cash': cash, 'correct': correct,
                    'confidence': proba, 'actual': 'YUKSELIS' if actual_direction == 1 else 'DUSUS'
                })

        elif signal == "SAT" and position == "LONG":
            revenue = shares * current_price
            cash += revenue

            correct = (actual_direction == 0)
            total_correct += correct
            total_trades += 1

            trades.append({
                'type': 'SAT', 'date': str(current_date), 'price': current_price,
                'shares': shares, 'cash': cash, 'correct': correct,
                'confidence': proba, 'actual': 'YUKSELIS' if actual_direction == 1 else 'DUSUS',
                'profit': revenue - (shares * entry_price)
            })

            shares = 0
            position = None

        port_val = cash + (shares * current_price)
        portfolio_history.append({
            'date': str(current_date), 'value': port_val, 'price': current_price,
            'signal': signal, 'position': position
        })

    if (idx + 1) % 3 == 0:
        print(f"  Donem {idx+1}/{len(windows)} tamamlandi...")

# Sonuclar
final_val = cash + (shares * portfolio_history[-1]['price']) if portfolio_history else cash
initial = 40000

print(f"\n{'='*70}")
print("SIMULASYON SONUCLARI")
print('='*70)
print(f"Baslangic: {initial:,.0f} TL")
print(f"Bitis:     {final_val:,.0f} TL")
print(f"Getiri:    %{((final_val-initial)/initial*100):+.2f}")

# Buy&Hold
bh_shares = int(40000 / gmstr['Close'].iloc[-min_len])
bh_val = bh_shares * gmstr['Close'].iloc[-1]
print(f"Buy&Hold:  {bh_val:,.0f} TL (%{((bh_val-initial)/initial*100):+.2f})")
print()

# Islem istatistikleri
print(f"Toplam Islem: {total_trades}")
print(f"Dogru: {total_correct}")
print(f"Yanlis: {total_trades - total_correct}")
print(f"Islem Basari: %{(total_correct/total_trades*100 if total_trades else 0):.1f}")
print()

# Tum islemler
if trades:
    print(f"{'='*70}")
    print("TUM ISLEMLER")
    print('='*70)
    print(f"{'TUR':<6} {'TARIH':<22} {'FIYAT':>10} {'GUVEN':>8} {'GERCEK':<10} {'SONUC':<8} {'KAR/ZARAR':>12}")
    print('-'*70)

    for t in trades:
        sonuc = "DOGRU" if t['correct'] else "YANLIS"
        kar = t.get('profit', 0)
        kar_str = f"{kar:+,.0f}" if 'profit' in t else "-"
        print(f"{t['type']:<6} {t['date'][:22]:<22} {t['price']:>10.2f} {t['confidence']*100:>7.1f}% {t['actual']:<10} {sonuc:<8} {kar_str:>12}")

# Portfoy trendi
print(f"\n{'='*70}")
print("AYLIK PORTFOY DEGERI")
print('='*70)
import pandas as pd
ph_df = pd.DataFrame(portfolio_history)
ph_df['date'] = pd.to_datetime(ph_df['date'])
monthly = ph_df.groupby(ph_df['date'].dt.to_period('M')).last()
for period, row in monthly.iterrows():
    print(f"{str(period):<10} | Portfoy: {row['value']:>12,.0f} TL | Fiyat: {row['price']:>8.2f}")
