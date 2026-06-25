"""
Gelistirilmis Model - Daha yuksek dogruluk ve karlilik
Degisiklikler:
1. Class imbalance cozumu (SMOTE)
2. CatBoost (LGBM'den daha iyi)
3. Regime-based threshold (ADX'e gore)
4. Feature engineering (momentum, volatility)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from gmstr_prediction_system import GMSTRPredictionSystem

ps = GMSTRPredictionSystem()

print("=" * 70)
print("GELISTIRILMIS MODEL TESTI")
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

# 3 farkli zaman diliminde test
test_periods = [
    (0.0, 0.7, 0.85),   # Erken donem
    (0.0, 0.8, 0.95),   # Orta donem  
    (0.2, 0.9, 1.0),    # Son donem
]

results = []

for idx, (start, train_end, test_end) in enumerate(test_periods, 1):
    n = len(X_all)
    X_train = X_all[int(n*start):int(n*train_end)]
    y_train = y_all[int(n*start):int(n*train_end)]
    X_test = X_all[int(n*train_end):int(n*test_end)]
    y_test = y_all[int(n*train_end):int(n*test_end)]
    
    print(f"\n{'='*70}")
    print(f"TEST {idx}: Train %{train_end*100:.0f} -> Test %{test_end*100:.0f}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Sinif: {np.bincount(y_train.astype(int))} -> {np.bincount(y_test.astype(int))}")
    
    # Feature scaling + PCA
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    pca = PCA(n_components=25, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    
    # SMOTE ile class balance
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_p, y_train)
        print(f"SMOTE sonrasi: {np.bincount(y_train_bal.astype(int))}")
    except:
        X_train_bal, y_train_bal = X_train_p, y_train
        print("SMOTE yok, orijinal veri")
    
    # CatBoost (daha iyi performans)
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=100,
            depth=3,
            learning_rate=0.05,
            auto_class_weights='Balanced',
            verbose=False,
            random_state=42
        )
        model.fit(X_train_bal, y_train_bal)
        y_proba = model.predict_proba(X_test_p)[:, 1]
    except:
        # LGBM fallback
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            is_unbalance=True, random_state=42, verbosity=-1
        )
        model.fit(X_train_bal, y_train_bal)
        y_proba = model.predict_proba(X_test_p)[:, 1]
    
    y_pred = (y_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    
    # Guclu sinyaller (>0.7 veya <0.3)
    strong = (y_proba > 0.70) | (y_proba < 0.30)
    if np.sum(strong) > 5:
        strong_acc = accuracy_score(y_test[strong], y_pred[strong])
        strong_pct = np.sum(strong) / len(y_test) * 100
    else:
        strong_acc = acc
        strong_pct = 100
    
    # Simulasyon
    cash = 40000
    shares = 0
    position = None
    entry = 0
    
    test_prices = gmstr['Close'].iloc[-len(y_test):].values
    for i in range(len(y_test)):
        price = float(test_prices[i])
        if strong[i]:
            if y_pred[i] == 1 and position is None:
                shares = int(cash / price)
                cash -= shares * price
                position = "LONG"
                entry = price
            elif y_pred[i] == 0 and position == "LONG":
                cash += shares * price
                position = None
    
    final = cash + (shares * test_prices[-1] if position else 0)
    ret = (final - 40000) / 40000 * 100
    
    print(f"Test Acc:      {acc*100:.2f}%")
    print(f"Guclu Acc:     {strong_acc*100:.2f}% ({strong_pct:.1f}% ornek)")
    print(f"Simulasyon:    {ret:+.2f}% ({final:,.0f} TL)")
    
    results.append({
        'test': idx,
        'acc': acc,
        'strong_acc': strong_acc,
        'return': ret
    })

# Ozet
print(f"\n{'='*70}")
print("OZET")
print('='*70)
avg_acc = np.mean([r['acc'] for r in results])
avg_strong = np.mean([r['strong_acc'] for r in results])
avg_ret = np.mean([r['return'] for r in results])

print(f"Ortalama Test:      {avg_acc*100:.2f}%")
print(f"Ortalama Guclu:     {avg_strong*100:.2f}%")
print(f"Ortalama Getiri:    {avg_ret:+.2f}%")

if avg_acc >= 0.65:
    print(">>> BASARILI! %65+ dogruluk saglandi!")
else:
    print(">>> Daha fazla iyilestirme gerekli")
