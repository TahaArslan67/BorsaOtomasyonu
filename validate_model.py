"""
Model overfitting ve ezber kontrolu
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from gmstr_prediction_system import GMSTRPredictionSystem

ps = GMSTRPredictionSystem()

print("=" * 70)
print("MODEL VALIDASYON - Overfitting Kontrolu")
print("=" * 70)

# Veri cek
gmstr = ps.fetch_gmstr_data(period="2y")
market = ps.fetch_market_data()

X = ps.create_features(gmstr, market)
y = ps.create_labels(gmstr)

min_len = min(len(X), len(y))
X, y = X[:min_len], y[:min_len]
mask = y != 2
X, y = X[mask], y[mask]

# Walk-forward split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
print(f"Sinif dagilimi (train): {np.bincount(y_train.astype(int))}")
print(f"Sinif dagilimi (test):  {np.bincount(y_test.astype(int))}")

# Feature selection
from sklearn.ensemble import RandomForestClassifier
rf_sel = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
rf_sel.fit(X_train, y_train)
importances = rf_sel.feature_importances_
top_idx = np.argsort(importances)[-50:]
X_train_sel = X_train[:, top_idx]
X_test_sel = X_test[:, top_idx]

# 1. Train vs Test accuracy
rf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_split=30,
                            min_samples_leaf=15, random_state=42, n_jobs=-1)
rf.fit(X_train_sel, y_train)

train_acc = accuracy_score(y_train, rf.predict(X_train_sel))
test_acc = accuracy_score(y_test, rf.predict(X_test_sel))

print(f"\n{'='*70}")
print("1. EZBER KONTROLU")
print('='*70)
print(f"Train Accuracy:  {train_acc*100:.2f}%")
print(f"Test Accuracy:   {test_acc*100:.2f}%")
print(f"Fark:            {train_acc*100 - test_acc*100:.2f} puan")

if train_acc - test_acc > 0.15:
    print("UYARI: Asiri ezberleme (overfitting) var!")
elif train_acc - test_acc > 0.05:
    print("Dikkat: Hafif overfitting var")
else:
    print("OK: Ezberleme yok, genelleme iyi")

# 2. TimeSeries Cross-Validation
print(f"\n{'='*70}")
print("2. ZAMAN SERISI CROSS-VALIDATION")
print('='*70)

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X[train_idx][:, top_idx], X[val_idx][:, top_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    rf_cv = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    rf_cv.fit(X_tr, y_tr)
    score = accuracy_score(y_val, rf_cv.predict(X_val))
    cv_scores.append(score)
    print(f"Fold {fold+1}: {score*100:.2f}% | Train: {len(X_tr)} | Val: {len(X_val)}")

print(f"\nCV Ortalama: {np.mean(cv_scores)*100:.2f}%")
print(f"CV Std:      {np.std(cv_scores)*100:.2f}%")

if np.std(cv_scores) > 0.15:
    print("UYARI: Yüksek varyans, model kararsiz!")

# 3. Confusion Matrix
print(f"\n{'='*70}")
print("3. CONFUSION MATRIX")
print('='*70)

y_pred = rf.predict(X_test_sel)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nDogru Tahminler: {cm[0,0] + cm[1,1]}")
print(f"Yanlis Tahminler: {cm[0,1] + cm[1,0]}")

# 4. Tahmin probability dagilimi
probas = rf.predict_proba(X_test_sel)[:, 1]
print(f"\n{'='*70}")
print("4. PROBABILITY DAGILIMI (ezber kontrolu)")
print('='*70)
print(f"Min:    {probas.min():.4f}")
print(f"Max:    {probas.max():.4f}")
print(f"Mean:   {probas.mean():.4f}")
print(f"Std:    {probas.std():.4f}")

# Cok yaklasik 0 veya 1 ise ezber
extreme = ((probas < 0.1) | (probas > 0.9)).mean()
print(f"\nExtreme probas (<0.1 veya >0.9): %{extreme*100:.1f}")

if extreme > 0.7:
    print("UYARI: Tahminler asiri keskin, ezber olabilir!")
else:
    print("OK: Tahminler dengeli dagilmis")

# 5. Farkli zaman dilimleri
print(f"\n{'='*70}")
print("5. FARKLI TEST DONEMLERI")
print('='*70)

for test_ratio in [0.1, 0.2, 0.3]:
    split_pt = int(len(X) * (1 - test_ratio))
    X_tr2, X_te2 = X[:split_pt][:, top_idx], X[split_pt:][:, top_idx]
    y_tr2, y_te2 = y[:split_pt], y[split_pt:]
    
    rf2 = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    rf2.fit(X_tr2, y_tr2)
    acc2 = accuracy_score(y_te2, rf2.predict(X_te2))
    print(f"Son %{test_ratio*100:.0f} test: {acc2*100:.2f}%")

print(f"\n{'='*70}")
print("SONUC")
print('='*70)

if train_acc - test_acc > 0.1 or extreme > 0.7:
    print("Model EZBERLEMIS olabilir. Guven DUSUK.")
    print("Oneri: max_depth dusur, min_samples_leaf artir")
else:
    print("Model genelleme iyi. Guven YUKSEK.")
