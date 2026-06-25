"""
40,000 TL ile 1 aylik GMSTR simulasyonu
"""
import numpy as np
import pandas as pd
from gmstr_prediction_system import GMSTRPredictionSystem

ps = GMSTRPredictionSystem()

print("=" * 60)
print("GMSTR 1 AY SIMULASYON - 40,000 TL Baslangic")
print("=" * 60)

# Veri cek
gmstr_data = ps.fetch_gmstr_data(period="2y")
market_data = ps.fetch_market_data()

# Model egit (eger yoksa)
if ps.model is None:
    ps.train_model()

# Son 1 aylik veriyi al
one_month_ago = gmstr_data.index[-1] - pd.Timedelta(days=30)
test_data = gmstr_data[gmstr_data.index >= one_month_ago].copy()

print(f"Test donemi: {test_data.index[0]} -> {test_data.index[-1]}")
print(f"Toplam bar: {len(test_data)}")

# Ozellikleri olustur (tam veri uzerinden, sonra son 1 ayi ayir)
X_all = ps.create_features(gmstr_data, market_data)

# Son 1 aylik ornekleri bul
n_test = len(test_data)
n_features = len(X_all)
n_start = n_features - n_test

X_test = X_all[n_start:]

# Simulasyon
initial_capital = 40000
cash = initial_capital
shares = 0
portfolio_value = initial_capital

# Pozisyon durumu
position = None  # None, 'LONG', 'SHORT'
entry_price = 0

# Istatistikler
trades = []
daily_values = []

print("\n" + "-" * 60)
print("SIMULASYON BASLIYOR...")
print("-" * 60)

for i in range(len(X_test)):
    current_price = test_data['Close'].iloc[i]
    timestamp = test_data.index[i]
    
    # Model tahmini
    features = X_test[i].reshape(1, -1)
    
    # Feature selection uygula
    if isinstance(ps.model, dict) and 'feature_indices' in ps.model:
        feat_idx = np.array(ps.model['feature_indices'])
        features = features[:, feat_idx]
    
    # Tahmin
    if isinstance(ps.model, dict) and 'rf' in ps.model:
        ensemble = ps.model
        w_rf, w_xgb, w_lgb = ensemble.get('weights', [1/3, 1/3, 1/3])
        rf_p = ensemble['rf'].predict_proba(features)[0][1]
        xgb_p = ensemble['xgb'].predict_proba(features)[0][1] if ensemble.get('xgb') else rf_p
        lgb_p = ensemble['lgb'].predict_proba(features)[0][1] if ensemble.get('lgb') else rf_p
        
        # Stacking
        if ensemble.get('meta') and ensemble.get('use_stacking'):
            meta_in = np.array([[rf_p, xgb_p, lgb_p]])
            proba = ensemble['meta'].predict_proba(meta_in)[0][1]
        else:
            proba = w_rf * rf_p + w_xgb * xgb_p + w_lgb * lgb_p
    else:
        proba = ps.model.predict_proba(features)[0][1]
    
    confidence = max(proba, 1 - proba)
    prediction = 1 if proba > 0.5 else 0
    
    # Kalite filtresi
    if abs(proba - 0.5) > 0.15:  # Guclu tahmin
        signal = "AL" if prediction == 1 else "SAT"
    else:
        signal = "BEKLE"
    
    # Islem mantigi (sadece LONG - spot hisse)
    if signal == "AL" and position is None:
        # Al
        shares = int(cash / current_price)
        cost = shares * current_price
        if shares > 0:
            cash -= cost
            position = "LONG"
            entry_price = current_price
            trades.append({
                'type': 'AL', 'time': timestamp, 'price': current_price,
                'shares': shares, 'cash': cash
            })
    
    elif signal == "SAT" and position == "LONG":
        # Sat
        revenue = shares * current_price
        cash += revenue
        shares = 0
        position = None
        trades.append({
            'type': 'SAT', 'time': timestamp, 'price': current_price,
            'shares': 0, 'cash': cash
        })
    
    # Portfoy degeri
    portfolio_value = cash + (shares * current_price)
    daily_values.append({'time': timestamp, 'value': portfolio_value, 'price': current_price})

# Sonuclar
final_price = test_data['Close'].iloc[-1]
final_value = cash + (shares * final_price)

# Eger hala pozisyon varsa, son fiyattan sat
if shares > 0:
    print(f"\n*** Simulasyon sonunda {shares} lot acik pozisyon kaldi ***")

# Buy&Hold karsilastirmasi
buyhold_shares = int(initial_capital / test_data['Close'].iloc[0])
buyhold_value = buyhold_shares * final_price

# Istatistikler
df_vals = pd.DataFrame(daily_values)
returns = df_vals['value'].pct_change().dropna()

if len(returns) > 0:
    total_return = (final_value - initial_capital) / initial_capital * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = ((df_vals['value'].cummax() - df_vals['value']) / df_vals['value'].cummax()).max() * 100
else:
    total_return = 0
    sharpe = 0
    max_dd = 0

# Islem ozeti
buy_signals = sum(1 for t in trades if t['type'] == 'AL')
sell_signals = sum(1 for t in trades if t['type'] == 'SAT')

print(f"\n{'=' * 60}")
print(f"SIMULASYON SONUCLARI")
print(f"{'=' * 60}")
print(f"Baslangic Sermaye: {initial_capital:,.0f} TL")
print(f"Bitis Portfoy Degeri: {final_value:,.0f} TL")
print(f"Toplam Getiri: %{total_return:+.2f}")
print(f"Buy&Hold Getiri: %{((buyhold_value - initial_capital) / initial_capital * 100):+.2f}")
print(f"\nIslem Sayisi:")
print(f"  AL (Yuk): {buy_signals}")
print(f"  SAT (Dus): {sell_signals}")
print(f"\nRisk Metrikleri:")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  Max Drawdown: %{max_dd:.2f}")

# Detayli gunluk log
print(f"\n{'-' * 60}")
print(f"DETAYLI GUNLUK LOG (Son 7 gun)")
print(f"{'-' * 60}")
for dv in daily_values[-7:]:
    print(f"{dv['time'].strftime('%Y-%m-%d %H:%M')} | Fiyat: {dv['price']:>8.2f} | Portfoy: {dv['value']:>10,.0f} TL")

# Tum islemler
if trades:
    print(f"\n{'-' * 60}")
    print(f"TUM ISLEMLER")
    print(f"{'-' * 60}")
    for t in trades:
        print(f"{t['type']:>3} | {t['time'].strftime('%Y-%m-%d %H:%M')} | Fiyat: {t['price']:>8.2f} | Nakit: {t['cash']:>10,.0f} TL")
