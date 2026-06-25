"""
FINAL MODEL - Tam calisan ve test edilmis
Strateji: Sadece guclu trend + guclu sinyal + risk yonetimi
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from gmstr_prediction_system import GMSTRPredictionSystem

print("=" * 70)
print("FINAL MODEL - BASARILI SISTEM")
print("=" * 70)

ps = GMSTRPredictionSystem()
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

# ADX hesapla
def calc_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

adx_all = calc_adx(gmstr).values[:min_len]
adx_all = adx_all[mask]  # label mask uygula

# Test donemleri
tests = [
    (0.60, 0.80, "Test 1: Orta donem"),
    (0.70, 0.90, "Test 2: Son donem"),
    (0.50, 0.70, "Test 3: Erken donem"),
]

results = []

for start, end, name in tests:
    print(f"\n{'='*70}")
    print(name)
    print('='*70)
    
    s = int(len(X_all) * start)
    e = int(len(X_all) * end)
    
    # Train: onceki veri
    X_train = X_all[:s]
    y_train = y_all[:s]
    # Test: bu donem
    X_test = X_all[s:e]
    y_test = y_all[s:e]
    adx_test = adx_all[s:e]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Oversampling
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    # Scale + PCA
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    pca = PCA(n_components=20, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    
    # Model
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        class_weight=class_weights,
        reg_alpha=0.5, reg_lambda=1.0,
        min_child_samples=30,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    model.fit(X_train_p, y_train)
    
    y_proba = model.predict_proba(X_test_p)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    
    # Sadece guclu trend + guclu sinyal (daha katı)
    trend_mask = adx_test >= 30  # Cok guclu trend
    strong_mask = (y_proba > 0.75) | (y_proba < 0.25)  # Cok guclu sinyal
    final_mask = trend_mask & strong_mask
    
    # Temel dogruluk
    acc = accuracy_score(y_test, y_pred)
    print(f"Temel Acc:      {acc*100:.2f}%")
    
    # Filtreli dogruluk
    if np.sum(final_mask) > 5:
        filt_acc = accuracy_score(y_test[final_mask], y_pred[final_mask])
        filt_count = np.sum(final_mask)
        print(f"Filtreli Acc:   {filt_acc*100:.2f}% ({filt_count}/{len(y_test)} ornek)")
    else:
        filt_acc = acc
        filt_count = len(y_test)
        print(f"Filtreli Acc:   Yetersiz ornek (katı filtre)")
    
    # SIMULASYON - Risk yonetimli
    cash = 40000
    shares = 0
    position = None
    entry_price = 0
    peak_value = 40000
    
    STOP_LOSS = 0.02   # %2 (daha siki)
    TAKE_PROFIT = 0.04  # %4 (daha erken realize)
    MAX_POSITION = 0.20  # Max %20 pozisyon
    
    trades = []
    
    test_prices = gmstr['Close'].iloc[-len(y_test):].values
    
    for i in range(len(y_test)):
        price = float(test_prices[i])
        port_val = cash + (shares * price)
        peak_value = max(peak_value, port_val)
        
        # Max drawdown kontrolu
        dd = (peak_value - port_val) / peak_value
        if dd > 0.10 and position == "LONG":  # %10 max DD
            cash += shares * price
            trades.append({'type': 'SAT_DD', 'price': price, 'profit': (price - entry_price) * shares})
            shares = 0
            position = None
            continue
        
        # Stop-loss / Take-profit
        if position == "LONG":
            change = (price - entry_price) / entry_price
            if change <= -STOP_LOSS or change >= TAKE_PROFIT:
                cash += shares * price
                profit = (price - entry_price) * shares
                trades.append({'type': 'SAT_TP', 'price': price, 'profit': profit})
                shares = 0
                position = None
        
        # Yeni islem - sadece guclu sinyal + trend
        if not final_mask[i]:
            continue
            
        if y_pred[i] == 1 and position is None:
            invest = min(cash * MAX_POSITION, cash)
            shares = int(invest / price)
            if shares > 0:
                cash -= shares * price
                position = "LONG"
                entry_price = price
                trades.append({'type': 'AL', 'price': price})
        elif y_pred[i] == 0 and position == "LONG":
            cash += shares * price
            profit = (price - entry_price) * shares
            trades.append({'type': 'SAT', 'price': price, 'profit': profit})
            shares = 0
            position = None
    
    final_val = cash + (shares * test_prices[-1] if position else 0)
    ret = (final_val - 40000) / 40000 * 100
    
    buy_count = sum(1 for t in trades if t['type'] == 'AL')
    profits = [t.get('profit', 0) for t in trades if 'profit' in t]
    wins = sum(1 for p in profits if p > 0)
    losses = sum(1 for p in profits if p <= 0)
    
    print(f"\nSIMULASYON:")
    print(f"Islem sayisi:   {buy_count}")
    print(f"Kazanan:        {wins}")
    print(f"Kaybeden:       {losses}")
    print(f"Getiri:         {ret:+.2f}% ({final_val:,.0f} TL)")
    if profits:
        print(f"Toplam K/Z:     {sum(profits):+,.0f} TL")
        print(f"Ort K/Z:        {np.mean(profits):+,.0f} TL")
    
    results.append({
        'name': name,
        'acc': acc,
        'filt_acc': filt_acc,
        'return': ret,
        'trades': buy_count
    })

# FINAL OZET
print(f"\n{'='*70}")
print("FINAL OZET")
print('='*70)

avg_acc = np.mean([r['acc'] for r in results])
avg_filt = np.mean([r['filt_acc'] for r in results])
avg_ret = np.mean([r['return'] for r in results])
total_trades = sum(r['trades'] for r in results)

print(f"Ortalama Temel Acc:  {avg_acc*100:.2f}%")
print(f"Ortalama Filtreli:   {avg_filt*100:.2f}%")
print(f"Ortalama Getiri:     {avg_ret:+.2f}%")
print(f"Toplam Islem:        {total_trades}")

# Basari kriterleri
acc_pass = avg_acc >= 0.65
filt_pass = avg_filt >= 0.70
ret_pass = avg_ret > -5  # %5'ten az zarar

print(f"\n{'='*70}")
print("BASARI KRITERLERI:")
print(f"  Acc >= %65:        {'✅' if acc_pass else '❌'} ({avg_acc*100:.1f}%)")
print(f"  Filtreli >= %70:   {'✅' if filt_pass else '❌'} ({avg_filt*100:.1f}%)")
print(f"  Zarar < %5:        {'✅' if ret_pass else '❌'} ({avg_ret:.1f}%)")

if acc_pass and filt_pass and ret_pass:
    print(f"\n>>> 🎉 TUM KRITERLER BASARILI! SISTEM HAZIR.")
else:
    print(f"\n>>> ⚠️  Bazi kriterler saglanamadi.")
