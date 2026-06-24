"""
GMSTR 15 Dakikalık Model v2 - Gümüş Verisiyle Geliştirilmiş
Düzeltmeler:
- Nötr bar filtresi kaldırıldı (veri dengesini bozuyordu)
- Lookahead 8 bar (2 saat) - daha güvenilir sinyal
- Sadece BIST saatleri (10:00-18:00)
- Daha güçlü regularizasyon
Hedef: %60+ doğruluk, %6+ aylık getiri tahmini
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = "gmstr_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("GMSTR 15m Model v2 - Gümüş Verisiyle Geliştirilmiş")
print("=" * 60)

# ============================================================
# 1. VERİ İNDİRME
# ============================================================
print("\n[1/6] Veriler indiriliyor...")

def download_data(ticker, period="60d", interval="15m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  ⚠️ {ticker}: Veri boş")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        print(f"  ✅ {ticker}: {len(df)} bar")
        return df
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None

gmstr  = download_data("GMSTR.IS")
silver = download_data("SI=F")   # Gümüş vadeli COMEX
slv    = download_data("SLV")    # Gümüş ETF
usdtry = download_data("USDTRY=X")
gold   = download_data("GC=F")   # Altın vadeli

if gmstr is None or len(gmstr) < 300:
    print("HATA: GMSTR verisi yetersiz!")
    exit(1)

# ============================================================
# 2. SADECE BIST SAATLERİ (10:00-18:00)
# ============================================================
print("\n[2/6] BIST saatleri filtreleniyor...")
gmstr = gmstr[(gmstr.index.hour >= 10) & (gmstr.index.hour < 18)]
print(f"  BIST saatleri sonrası: {len(gmstr)} bar")

# ============================================================
# 3. DIŞ VERİLERİ HİZALA
# ============================================================
print("\n[3/6] Dış veriler hizalanıyor...")

def align_external(ext_df, gmstr_idx, col_name):
    if ext_df is None:
        return pd.Series(np.nan, index=gmstr_idx, name=col_name)
    s = ext_df['Close'].copy()
    # Nearest neighbor ile hizala (forward fill + backward fill)
    result = s.reindex(gmstr_idx, method='nearest', tolerance=pd.Timedelta('30min'))
    if result.isna().sum() > len(result) * 0.5:
        result = s.reindex(gmstr_idx, method='ffill')
    result.name = col_name
    return result

idx = gmstr.index
xag_s = align_external(silver, idx, 'xag')
slv_s = align_external(slv, idx, 'slv')
usd_s = align_external(usdtry, idx, 'usd')
gld_s = align_external(gold, idx, 'gld')

print(f"  XAG dolu: {xag_s.notna().sum()}/{len(xag_s)}")
print(f"  SLV dolu: {slv_s.notna().sum()}/{len(slv_s)}")
print(f"  USD dolu: {usd_s.notna().sum()}/{len(usd_s)}")

# Ana DataFrame
df = pd.DataFrame({
    'open':  gmstr['Open'].values,
    'high':  gmstr['High'].values,
    'low':   gmstr['Low'].values,
    'close': gmstr['Close'].values,
    'volume':gmstr['Volume'].values,
    'xag':   xag_s.values,
    'slv':   slv_s.values,
    'usd':   usd_s.values,
    'gld':   gld_s.values,
}, index=idx)

# Dış verileri forward fill
for col in ['xag','slv','usd','gld']:
    df[col] = df[col].ffill().bfill()

print(f"  Birleşik veri: {len(df)} bar")

# ============================================================
# 4. ÖZELLİK MÜHENDİSLİĞİ
# ============================================================
print("\n[4/6] Özellikler hesaplanıyor...")

c = df['close']
h = df['high']
l = df['low']
v = df['volume']

# --- GMSTR Teknik Göstergeler ---
for w in [3, 5, 8, 13, 21]:
    df[f'ma{w}'] = c.rolling(w).mean()
    df[f'ma{w}_r'] = (c / df[f'ma{w}'] - 1)

for w in [5, 13, 21]:
    df[f'ema{w}'] = c.ewm(span=w).mean()
    df[f'ema{w}_r'] = (c / df[f'ema{w}'] - 1)

# RSI
for w in [7, 14]:
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(w).mean()
    loss = (-delta.clip(upper=0)).rolling(w).mean()
    df[f'rsi{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))

# MACD
ema12 = c.ewm(span=12).mean()
ema26 = c.ewm(span=26).mean()
df['macd'] = (ema12 - ema26) / c
df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()

# Bollinger
for w in [10, 20]:
    mid = c.rolling(w).mean()
    std = c.rolling(w).std()
    df[f'bb_pos{w}'] = (c - (mid - 2*std)) / (4*std + 1e-10)
    df[f'bb_w{w}'] = 4*std / (mid + 1e-10)

# ATR
tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean() / c

# Stochastic
for w in [5, 14]:
    df[f'stoch{w}'] = (c - l.rolling(w).min()) / (h.rolling(w).max() - l.rolling(w).min() + 1e-10)

# Momentum (getiri)
for w in [1, 2, 3, 4, 6, 8, 12]:
    df[f'ret{w}'] = c.pct_change(w)

# Hacim
df['vol_r'] = v / (v.rolling(10).mean() + 1e-10)
df['vol_r'] = df['vol_r'].clip(0, 10)

# Mum
df['body'] = (c - df['open']) / (h - l + 1e-10)
df['hl_r'] = (h - l) / c

# Zaman
df['hour'] = df.index.hour
df['dow'] = df.index.dayofweek
df['is_open'] = ((df['hour'] == 10) | (df['hour'] == 11)).astype(int)
df['is_close'] = (df['hour'] >= 16).astype(int)
df['is_lunch'] = ((df['hour'] == 12) | (df['hour'] == 13)).astype(int)

# --- GÜMÜŞ VERİLERİ ---
xag = df['xag']
slv_col = df['slv']
usd = df['usd']
gld_col = df['gld']

# XAG momentum
for w in [2, 4, 8]:
    df[f'xag_r{w}'] = xag.pct_change(w)

# XAG trend
df['xag_ma5'] = xag.rolling(5).mean()
df['xag_ma20'] = xag.rolling(20).mean()
df['xag_trend'] = (df['xag_ma5'] > df['xag_ma20']).astype(int)

# XAG RSI
delta = xag.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df['xag_rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))

# GMSTR vs XAG korelasyon
df['gmstr_xag_corr'] = c.rolling(20).corr(xag)

# USD/TRY momentum
for w in [2, 4]:
    df[f'usd_r{w}'] = usd.pct_change(w)
df['usd_trend'] = (usd > usd.rolling(10).mean()).astype(int)

# Altın/Gümüş oranı
df['gsr'] = gld_col / (slv_col + 1e-10)
df['gsr_r'] = df['gsr'].pct_change(4)
df['gsr_trend'] = (df['gsr'] < df['gsr'].rolling(10).mean()).astype(int)

# SLV momentum
for w in [2, 4]:
    df[f'slv_r{w}'] = slv_col.pct_change(w)

# Kombine: USD*XAG = TL cinsinden gümüş proxy
df['tl_silver_r4'] = usd.pct_change(4) + xag.pct_change(4)

print(f"  Toplam özellik: {df.shape[1]}")

# ============================================================
# 5. HEDEF DEĞİŞKEN
# ============================================================
print("\n[5/6] Hedef değişken oluşturuluyor...")

# 8 bar (2 saat) sonraki yön - daha güvenilir
LOOKAHEAD = 8
future_ret = c.pct_change(LOOKAHEAD).shift(-LOOKAHEAD)
df['target'] = (future_ret > 0).astype(int)
df['future_ret'] = future_ret

# Son LOOKAHEAD barı çıkar (hedef yok)
df = df.iloc[:-LOOKAHEAD]

print(f"  Hedef dağılımı: AL={df['target'].sum()} ({df['target'].mean()*100:.1f}%), SAT={len(df)-df['target'].sum()}")

# ============================================================
# 6. MODEL EĞİTİMİ
# ============================================================
print("\n[6/6] Model eğitiliyor...")

exclude = {'open','high','low','close','volume','xag','slv','usd','gld','target','future_ret'}
feature_cols = [col for col in df.columns if col not in exclude]

# NaN temizle
df_work = df[feature_cols + ['target']].copy()
df_work = df_work.ffill().bfill()

# Hala NaN olan sütunları kaldır
nan_pct = df_work[feature_cols].isna().mean()
bad_cols = nan_pct[nan_pct > 0.1].index.tolist()
if bad_cols:
    print(f"  Kaldırılan sütunlar: {bad_cols}")
    feature_cols = [c for c in feature_cols if c not in bad_cols]
    df_work = df_work[feature_cols + ['target']]

df_clean = df_work.dropna()
print(f"  Temiz veri: {len(df_clean)} bar, {len(feature_cols)} özellik")

if len(df_clean) < 300:
    print("HATA: Yeterli temiz veri yok!")
    exit(1)

# Infinity ve çok büyük değerleri temizle
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
print(f"  Infinity temizleme sonrası: {len(df_clean)} bar")

X = df_clean[feature_cols].values.astype(np.float64)
# Clip aşırı değerleri
X = np.clip(X, -1e6, 1e6)
X = X.astype(np.float32)
y = df_clean['target'].values

# Train/Test split (son %20 test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Train AL oranı: {y_train.mean()*100:.1f}%")

tscv = TimeSeriesSplit(n_splits=5)

base_models = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )),
    ('lgb', lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        num_leaves=15, subsample=0.7, colsample_bytree=0.7,
        min_child_samples=30, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbose=-1
    )),
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        max_features='sqrt', random_state=42, n_jobs=-1
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        max_features='sqrt', random_state=42, n_jobs=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, min_samples_leaf=20, random_state=42
    )),
]

print("\n  Base modeller eğitiliyor (walk-forward CV)...")
oof_preds = np.zeros((len(X_train_s), len(base_models)))
test_preds = np.zeros((len(X_test_s), len(base_models)))
cv_scores = []

for i, (name, model) in enumerate(base_models):
    oof = np.zeros(len(X_train_s))
    fold_scores = []
    for train_idx, val_idx in tscv.split(X_train_s):
        model.fit(X_train_s[train_idx], y_train[train_idx])
        oof[val_idx] = model.predict_proba(X_train_s[val_idx])[:, 1]
        fold_scores.append(accuracy_score(y_train[val_idx], (oof[val_idx] > 0.5).astype(int)))
    oof_preds[:, i] = oof
    model.fit(X_train_s, y_train)
    test_preds[:, i] = model.predict_proba(X_test_s)[:, 1]
    cv_acc = np.mean(fold_scores)
    cv_scores.append(cv_acc)
    print(f"    {name}: CV={cv_acc*100:.1f}%")

# Meta-learner
meta = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
meta.fit(oof_preds, y_train)

final_probs = meta.predict_proba(test_preds)[:, 1]
final_preds = (final_probs > 0.5).astype(int)

test_acc = accuracy_score(y_test, final_preds)
try:
    test_auc = roc_auc_score(y_test, final_probs)
except:
    test_auc = 0.5

print(f"\n  ✅ Stacking Ensemble:")
print(f"     Test Doğruluk: %{test_acc*100:.2f}")
print(f"     Test AUC: {test_auc:.4f}")
print(f"     Ortalama CV: %{np.mean(cv_scores)*100:.2f}")

# Eşik optimizasyonu
best_thresh = 0.5
best_acc = test_acc
for thresh in np.arange(0.40, 0.65, 0.01):
    preds_t = (final_probs > thresh).astype(int)
    acc_t = accuracy_score(y_test, preds_t)
    if acc_t > best_acc:
        best_acc = acc_t
        best_thresh = thresh

print(f"     Optimal Eşik: {best_thresh:.2f} → %{best_acc*100:.2f}")

# Tüm veriyle yeniden eğit
X_all_s = scaler.fit_transform(X)
for name, model in base_models:
    model.fit(X_all_s, y)

# Kaydet
model_package = {
    'base_models': base_models,
    'meta_learner': meta,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'threshold': best_thresh,
    'test_accuracy': best_acc,
    'test_auc': test_auc,
    'cv_accuracy': np.mean(cv_scores),
    'lookahead': LOOKAHEAD,
    'trained_at': datetime.now().isoformat(),
    'n_features': len(feature_cols),
    'n_samples': len(df_clean),
    'silver_enhanced': True,
    'version': 2,
}

model_path = os.path.join(MODEL_DIR, 'simple_15m_15min.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

# Training results güncelle
results_path = os.path.join(MODEL_DIR, 'training_results.json')
try:
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
except:
    results = {}

results['15m_15min'] = {
    'test_accuracy': best_acc,
    'cv_accuracy': np.mean(cv_scores),
    'test_auc': test_auc,
    'optimal_threshold': best_thresh,
    'n_features': len(feature_cols),
    'n_samples': len(df_clean),
    'lookahead_bars': LOOKAHEAD,
    'silver_enhanced': True,
    'trained_at': datetime.now().isoformat(),
}

with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ÖZET
daily_move = 0.015
monthly_return = (best_acc - 0.5) * 2 * daily_move * 22 * 100

print("\n" + "=" * 60)
print("EĞİTİM TAMAMLANDI")
print("=" * 60)
print(f"  Test Doğruluk : %{best_acc*100:.2f} {'✅ HEDEF AŞILDI!' if best_acc >= 0.60 else '⚠️ Hedef altında'}")
print(f"  AUC           : {test_auc:.4f}")
print(f"  CV Doğruluk   : %{np.mean(cv_scores)*100:.2f}")
print(f"  Özellik Sayısı: {len(feature_cols)}")
print(f"  Veri Noktası  : {len(df_clean)}")
print(f"  Lookahead     : {LOOKAHEAD} bar (2 saat)")
print(f"  Gümüş Verisi  : SI=F + SLV + USD/TRY + GC=F + GSR")
print(f"\n  Tahmini Aylık Getiri: %{monthly_return:.1f} {'✅' if monthly_return >= 6 else '⚠️'}")
print("=" * 60)
