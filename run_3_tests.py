"""
3 kere walk-forward test
"""
import numpy as np
import pandas as pd
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
print("3 KERE WALK-FORWARD TEST")
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

# 3 farkli test donemi
splits = [
    (0.6, 0.8),   # Test 1: orta donem
    (0.7, 0.9),   # Test 2: son donem
    (0.5, 0.7),   # Test 3: erken donem
]

results = []

for test_idx, (train_end, test_end) in enumerate(splits, 1):
    print(f"\n{'='*70}")
    print(f"TEST {test_idx}: Train 0-{int(train_end*100)}% | Test {int(train_end*100)}-{int(test_end*100)}%")
    print('='*70)
    
    train_size = int(len(X_all) * train_end)
    test_size = int(len(X_all) * test_end)
    
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test = X_all[train_size:test_size]
    y_test = y_all[train_size:test_size]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Sinif dagilimi: {np.bincount(y_train.astype(int))} -> {np.bincount(y_test.astype(int))}")
    
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
    
    # Test
    y_pred = model.predict(X_test_p)
    y_proba = model.predict_proba(X_test_p)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    train_pred = model.predict(X_train_p)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Kalite filtresi (>0.75 veya <0.25)
    strong_mask = (y_proba > 0.75) | (y_proba < 0.25)
    if np.sum(strong_mask) > 10:
        strong_acc = accuracy_score(y_test[strong_mask], y_pred[strong_mask])
        strong_count = np.sum(strong_mask)
    else:
        strong_acc = acc
        strong_count = len(y_test)
    
    print(f"Train Acc: {train_acc*100:.2f}%")
    print(f"Test Acc:  {acc*100:.2f}%")
    print(f"Fark:      {train_acc*100 - acc*100:.2f} puan")
    print(f"Guclu Sinyal Acc: {strong_acc*100:.2f}% | {strong_count}/{len(y_test)} ornek")
    
    results.append({
        'test': test_idx,
        'train_acc': train_acc,
        'test_acc': acc,
        'diff': train_acc - acc,
        'strong_acc': strong_acc,
        'strong_count': strong_count
    })

# Ozet
print(f"\n{'='*70}")
print("OZET")
print('='*70)
avg_test = np.mean([r['test_acc'] for r in results])
avg_strong = np.mean([r['strong_acc'] for r in results])
avg_diff = np.mean([r['diff'] for r in results])

print(f"Ortalama Test Acc:    {avg_test*100:.2f}%")
print(f"Ortalama Guclu Acc:   {avg_strong*100:.2f}%")
print(f"Ortalama Train/Test:  {avg_diff*100:.2f} puan")
print()

if avg_test >= 0.65:
    print(">>> BASARILI! %65+ dogruluk saglandi!")
elif avg_test >= 0.60:
    print(">>> YAKIN! %60+ var, biraz daha iyilestirilebilir")
else:
    print(">>> DUSUK! Daha fazla iyilestirme gerekli")
