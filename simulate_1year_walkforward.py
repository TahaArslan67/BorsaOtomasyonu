"""
1 yillik walk-forward simulasyon
Her ay egitim, sonraki ay test
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gmstr_prediction_system import GMSTRPredictionSystem
import warnings
warnings.filterwarnings('ignore')

ps = GMSTRPredictionSystem()

print("=" * 70)
print("1 YILLIK WALK-FORWARD SIMULASYON")
print("Baslangic: 40,000 TL")
print("=" * 70)

# Veri cek
print("\nVeri cekiliyor...")
gmstr = ps.fetch_gmstr_data(period="2y")
market = ps.fetch_market_data()

X_all = ps.create_features(gmstr, market)
y_all = ps.create_labels(gmstr)

min_len = min(len(X_all), len(y_all))
X_all = X_all[:min_len]
y_all = y_all[:min_len]

# Tarih indeksi
dates = gmstr.index[-min_len:]

# Walk-forward: Her 200 bar egitim, sonraki 50 bar test (aylik yaklasik)
window_train = 2000  # ~8-9 ay
window_test = 200    # ~1 ay

cash = 40000
shares = 0
position = None  # None, 'LONG'
trades = []
portfolio_history = []

total_correct = 0
total_trades = 0

# Kac window var?
n_samples = len(X_all)
step = 200  # kaydirma adimi

windows = []
start = window_train
while start + window_test <= n_samples:
    windows.append((start - window_train, start, start + window_test))
    start += step

print(f"Toplam {len(windows)} donem test edilecek")
print(f"Her donem: {window_train} egitim + {window_test} test bar\n")

for idx, (train_start, train_end, test_end) in enumerate(windows):
    X_train = X_all[train_start:train_end]
    y_train = y_all[train_start:train_end]
    X_test = X_all[train_end:test_end]
    y_test = y_all[train_end:test_end]
    test_dates = dates[train_end:test_end]
    test_prices = gmstr['Close'].iloc[train_end:test_end].values

    # YATAY filtrele
    mask = y_train != 2
    X_train, y_train = X_train[mask], y_train[mask]

    if len(X_train) < 100 or len(X_test) == 0:
        continue

    # Feature selection
    rf_sel = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train, y_train)
    top_idx = np.argsort(rf_sel.feature_importances_)[-30:]
    X_train_sel = X_train[:, top_idx]
    X_test_sel = X_test[:, top_idx]

    # Model egit
    rf = RandomForestClassifier(
        n_estimators=80, max_depth=3, min_samples_split=100,
        min_samples_leaf=50, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sel, y_train)

    # Test doneminde her bar icin tahmin
    for i in range(len(X_test)):
        current_price = float(test_prices[i])
        current_date = test_dates[i]
        actual_direction = y_test[i]

        features = X_test_sel[i].reshape(1, -1)
        proba = rf.predict_proba(features)[0][1]
        prediction = 1 if proba > 0.5 else 0

        # Kalite filtresi (|proba-0.5| > 0.15)
        if abs(proba - 0.5) > 0.15:
            signal = "AL" if prediction == 1 else "SAT"
        else:
            signal = "BEKLE"

        # Islem
        trade_executed = False
        if signal == "AL" and position is None:
            shares = int(cash / current_price)
            if shares > 0:
                cost = shares * current_price
                cash -= cost
                position = "LONG"
                entry_price = current_price
                trade_executed = True

                # Dogru mu?
                correct = (actual_direction == 1)
                total_correct += correct
                total_trades += 1

                trades.append({
                    'type': 'AL', 'date': current_date, 'price': current_price,
                    'shares': shares, 'cash': cash, 'correct': correct,
                    'confidence': proba, 'actual': 'YUKSELIS' if actual_direction == 1 else 'DUSUS'
                })

        elif signal == "SAT" and position == "LONG":
            revenue = shares * current_price
            cash += revenue

            # Dogru mu? (SAT dedik, dustu mu gercekten?)
            correct = (actual_direction == 0)
            total_correct += correct
            total_trades += 1

            trades.append({
                'type': 'SAT', 'date': current_date, 'price': current_price,
                'shares': shares, 'cash': cash, 'correct': correct,
                'confidence': proba, 'actual': 'YUKSELIS' if actual_direction == 1 else 'DUSUS',
                'profit': revenue - (shares * entry_price)
            })

            shares = 0
            position = None

        # Portfoy degeri
        port_val = cash + (shares * current_price)
        portfolio_history.append({
            'date': current_date, 'value': port_val, 'price': current_price,
            'signal': signal, 'position': position
        })

    if (idx + 1) % 5 == 0:
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
print()

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
    print(f"{'TUR':<4} {'TARIH':<20} {'FIYAT':>8} {'GUVEN':>6} {'GERCEK':<8} {'SONUC':<8} {'KAR/ZARAR':>10}")
    print('-'*70)

    for t in trades:
        sonuc = "DOGRU" if t['correct'] else "YANLIS"
        kar = t.get('profit', 0)
        kar_str = f"{kar:+,.0f}" if 'profit' in t else "-"
        print(f"{t['type']:<4} {str(t['date'])[:19]:<20} {t['price']:>8.2f} {t['confidence']*100:>5.1f}% {t['actual']:<8} {sonuc:<8} {kar_str:>10}")

# Gunluk portfoy (son 20)
print(f"\n{'='*70}")
print("SON 20 GUN PORTFOY DEGERI")
print('='*70)
for ph in portfolio_history[-20:]:
    print(f"{str(ph['date'])[:16]:<16} | Fiyat: {ph['price']:>8.2f} | Portfoy: {ph['value']:>10,.0f} TL | Sinyal: {ph['signal']:<6}")
