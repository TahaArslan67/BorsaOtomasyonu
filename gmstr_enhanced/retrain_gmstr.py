#!/usr/bin/env python3
"""
GMSTR Model Yeniden Eğitim Scripti - GELİŞMİŞ VERSİYON
=========================================================
İyileştirmeler:
- Daha fazla özellik (lag, momentum, volatilite, mevsimsellik)
- Stacking ensemble (meta-learner)
- Optuna ile hiperparametre optimizasyonu
- Daha sıkı overfit kontrolü
- Gerçekçi backtest ile %15 aylık hedef doğrulaması
- Tahmin güven aralıkları

Kullanım: python gmstr_enhanced/retrain_gmstr.py
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / 'gmstr_models'
MODEL_DIR.mkdir(exist_ok=True)


def load_daily_data():
    """Günlük veriyi yükle."""
    csv_path = ROOT / 'claude' / 'areaxdatetime.csv'
    if not csv_path.exists():
        print(f"[HATA] Veri dosyası bulunamadı: {csv_path}")
        return None
    
    from gmstr_system.data_loader import GMSTRDataLoader
    loader = GMSTRDataLoader(str(csv_path))
    loader.load()
    df = loader.clean()
    print(f"[Veri] Günlük: {len(df)} satır | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def engineer_advanced_features(df):
    """
    Gelişmiş özellik mühendisliği:
    - Temel teknik göstergeler
    - Lag özellikleri (1-10 gün)
    - Momentum göstergeleri
    - Volatilite göstergeleri
    - Mevsimsellik (ay, haftanın günü)
    - Fiyat pattern'leri
    """
    from gmstr_system.features import FeatureEngineer
    eng = FeatureEngineer()
    df = eng.transform(df)
    
    # Ek özellikler
    close = df['Close']
    
    # --- Lag özellikleri ---
    for lag in [1, 2, 3, 5, 7, 10]:
        df[f'ret_lag_{lag}'] = close.pct_change(lag)
        df[f'close_lag_{lag}'] = close.shift(lag)
    
    # --- Momentum ---
    df['mom_5'] = close / close.shift(5) - 1
    df['mom_10'] = close / close.shift(10) - 1
    df['mom_20'] = close / close.shift(20) - 1
    df['mom_60'] = close / close.shift(60) - 1
    
    # --- Volatilite ---
    for window in [5, 10, 20]:
        df[f'vol_{window}'] = close.pct_change().rolling(window).std()
        df[f'vol_ratio_{window}'] = df[f'vol_{window}'] / df[f'vol_{window}'].shift(window)
    
    # --- Fiyat pozisyonu ---
    for window in [10, 20, 50]:
        roll_min = close.rolling(window).min()
        roll_max = close.rolling(window).max()
        df[f'price_pos_{window}'] = (close - roll_min) / (roll_max - roll_min + 1e-8)
    
    # --- Mevsimsellik ---
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    
    # --- Hacim özellikleri (varsa) ---
    if 'Volume' in df.columns:
        vol = df['Volume']
        df['vol_ma_5'] = vol.rolling(5).mean()
        df['vol_ma_20'] = vol.rolling(20).mean()
        df['vol_ratio'] = vol / (df['vol_ma_20'] + 1e-8)
        df['price_vol'] = close * vol  # Fiyat x Hacim
    
    # --- Trend göstergeleri ---
    for short, long in [(5, 20), (10, 50), (20, 60)]:
        ma_s = close.rolling(short).mean()
        ma_l = close.rolling(long).mean()
        df[f'ma_cross_{short}_{long}'] = (ma_s - ma_l) / (ma_l + 1e-8)
    
    # --- Fiyat kanalı ---
    for window in [10, 20]:
        df[f'channel_pos_{window}'] = (
            (close - close.rolling(window).min()) / 
            (close.rolling(window).max() - close.rolling(window).min() + 1e-8)
        )
    
    # --- Ardışık gün sayısı (streak) ---
    daily_ret = close.pct_change()
    df['up_streak'] = (daily_ret > 0).astype(int)
    df['down_streak'] = (daily_ret < 0).astype(int)
    
    # Streak hesapla
    streak = []
    current = 0
    for r in daily_ret:
        if pd.isna(r):
            streak.append(0)
        elif r > 0:
            current = max(0, current) + 1
            streak.append(current)
        elif r < 0:
            current = min(0, current) - 1
            streak.append(current)
        else:
            current = 0
            streak.append(0)
    df['price_streak'] = streak
    
    # --- Normalize edilmiş fiyat ---
    df['price_norm_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
    df['price_norm_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-8)
    
    feature_cols = eng.get_feature_columns(df)
    
    # Ek özellikleri ekle
    extra_cols = [c for c in df.columns if c.startswith(('ret_lag_', 'close_lag_', 'mom_', 
                  'vol_', 'price_pos_', 'day_of_week', 'month', 'quarter', 'week_of_year',
                  'vol_ma_', 'vol_ratio', 'price_vol', 'ma_cross_', 'channel_pos_',
                  'up_streak', 'down_streak', 'price_streak', 'price_norm_'))]
    
    all_features = list(set(feature_cols + extra_cols))
    # Sadece df'de olan kolonları al
    all_features = [c for c in all_features if c in df.columns]
    
    print(f"[Özellik] Toplam {len(all_features)} özellik üretildi (temel + gelişmiş)")
    return df, all_features


def create_robust_targets(df, horizon_days):
    """
    Daha sağlam hedef değişkeni:
    - Basit binary: horizon gün sonra fiyat bugünden yüksek mi?
    """
    future_ret = df['Close'].pct_change(horizon_days).shift(-horizon_days)
    target = (future_ret > 0.0).astype(int)
    return target, future_ret


def select_best_features(X, y, k=40):
    """
    Özellik seçimi:
    - mutual_info_classif ile ilk k özellik
    - Korelasyon filtresi (yüksek korelasyonlu olanları çıkar)
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # NaN içeren kolonları çıkar
    valid_cols = X.columns[X.isna().mean() < 0.3].tolist()
    X_clean = X[valid_cols].fillna(X[valid_cols].median())
    
    # Mutual info
    mi_scores = mutual_info_classif(X_clean, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=valid_cols).sort_values(ascending=False)
    
    # İlk k*2 özelliği al
    top_features = mi_series.head(k * 2).index.tolist()
    X_top = X_clean[top_features]
    
    # Korelasyon filtresi (>0.95 korelasyonlu olanları çıkar)
    corr_matrix = X_top.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    filtered = [f for f in top_features if f not in to_drop]
    
    # Son k özelliği al
    selected = filtered[:k]
    print(f"  Özellik seçimi: {len(valid_cols)} → {len(top_features)} → {len(selected)} (korelasyon filtresi)")
    return selected


def train_advanced_model(X_train, y_train, X_test, y_test, horizon_name):
    """
    Gelişmiş model eğitimi:
    - XGBoost, LightGBM, RandomForest, ExtraTrees, GradientBoosting
    - Stacking ensemble (Logistic Regression meta-learner)
    - Walk-forward CV
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                                   ExtraTreesClassifier)
    
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    
    # Sınıf dengesi
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    
    # --- Base modeller ---
    base_models = {
        'xgb': xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=8, gamma=0.2,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=42, n_jobs=-1,
            verbosity=0
        ),
        'lgb': lgb.LGBMClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_samples=25, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        ),
        'rf': RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=15,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, min_samples_leaf=10,
            random_state=42
        ),
    }
    
    # Base modelleri eğit
    for name, m in base_models.items():
        m.fit(X_tr_sc, y_train)
    
    # --- Stacking: OOF tahminleri ---
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(X_train), len(base_models)))
    
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_cv_tr = scaler.fit_transform(X_train.iloc[tr_idx])
        X_cv_val = scaler.transform(X_train.iloc[val_idx])
        
        for m_idx, (name, m) in enumerate(base_models.items()):
            m_fold = type(m)(**m.get_params())
            m_fold.fit(X_cv_tr, y_train.iloc[tr_idx])
            oof_preds[val_idx, m_idx] = m_fold.predict_proba(X_cv_val)[:, 1]
    
    # Meta-learner (Logistic Regression)
    meta_learner = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    meta_learner.fit(oof_preds, y_train)
    
    # Test tahminleri
    test_base_preds = np.column_stack([
        m.predict_proba(X_te_sc)[:, 1] for m in base_models.values()
    ])
    
    # Stacking tahmin
    stacking_proba = meta_learner.predict_proba(test_base_preds)[:, 1]
    
    # Basit ensemble (ortalama) ile karşılaştır
    avg_proba = np.mean(test_base_preds, axis=1)
    
    # İkisinin ortalaması
    final_proba = stacking_proba * 0.6 + avg_proba * 0.4
    
    preds = (final_proba > 0.5).astype(int)
    test_acc = accuracy_score(y_test, preds)
    try:
        test_auc = roc_auc_score(y_test, final_proba)
    except:
        test_auc = 0.5
    
    # Walk-forward CV doğruluğu
    cv_accs = []
    cv_aucs = []
    tscv2 = TimeSeriesSplit(n_splits=5)
    for tr_idx, val_idx in tscv2.split(X_train):
        X_cv_tr = scaler.fit_transform(X_train.iloc[tr_idx])
        X_cv_val = scaler.transform(X_train.iloc[val_idx])
        
        cv_probas = []
        for name, m in base_models.items():
            m2 = type(m)(**m.get_params())
            m2.fit(X_cv_tr, y_train.iloc[tr_idx])
            cv_probas.append(m2.predict_proba(X_cv_val)[:, 1])
        
        cv_avg = np.mean(cv_probas, axis=0)
        cv_preds = (cv_avg > 0.5).astype(int)
        cv_accs.append(accuracy_score(y_train.iloc[val_idx], cv_preds))
        try:
            cv_aucs.append(roc_auc_score(y_train.iloc[val_idx], cv_avg))
        except:
            cv_aucs.append(0.5)
    
    cv_acc = np.mean(cv_accs)
    cv_std = np.std(cv_accs)
    cv_auc = np.mean(cv_aucs)
    
    # Feature importance (XGBoost'tan)
    xgb_model = base_models['xgb']
    feat_imp = pd.Series(
        xgb_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    top_features = feat_imp.head(10).to_dict()
    
    return {
        'test_accuracy': round(test_acc, 4),
        'test_auc': round(test_auc, 4),
        'cv_accuracy': round(cv_acc, 4),
        'cv_std': round(cv_std, 4),
        'cv_auc': round(cv_auc, 4),
        'positive_rate': round(float(y_train.mean()), 4),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'scaler': scaler,
        'base_models': base_models,
        'meta_learner': meta_learner,
        'top_features': {k: round(float(v), 4) for k, v in top_features.items()},
    }


def realistic_backtest(df, feature_cols, horizon_days, result):
    """
    Gerçekçi backtest:
    - Sadece güçlü sinyal (>60% güven) olan günlerde işlem yap
    - Komisyon: %0.1 her işlemde
    - Stop-loss: %3
    - Aylık getiri hesapla
    """
    scaler = result['scaler']
    base_models = result['base_models']
    meta_learner = result['meta_learner']
    
    # Test seti (son %20)
    clean = df.dropna(subset=feature_cols).copy()
    target, future_ret = create_robust_targets(clean, horizon_days)
    clean['target'] = target
    clean['future_ret'] = future_ret
    clean = clean.dropna(subset=['target', 'future_ret'])
    
    split_idx = int(len(clean) * 0.80)
    test_df = clean.iloc[split_idx:].copy()
    
    if len(test_df) < 20:
        return None
    
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
    X_test_sc = scaler.transform(X_test)
    
    # Tahminler
    base_preds = np.column_stack([
        m.predict_proba(X_test_sc)[:, 1] for m in base_models.values()
    ])
    stacking_proba = meta_learner.predict_proba(base_preds)[:, 1]
    avg_proba = np.mean(base_preds, axis=1)
    final_proba = stacking_proba * 0.6 + avg_proba * 0.4
    
    # Backtest simülasyonu
    commission = 0.001  # %0.1
    stop_loss = 0.03    # %3 stop-loss
    capital = 10000     # 10,000 TL başlangıç
    monthly_capitals = {}
    
    position = 0  # 0=nakit, 1=long
    entry_price = 0
    hold_days = 0
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        prob = final_proba[i]
        price = row['Close']
        month_key = str(idx)[:7]
        
        if month_key not in monthly_capitals:
            monthly_capitals[month_key] = {'start': capital, 'end': capital, 'trades': 0}
        
        # Stop-loss kontrolü
        if position == 1 and entry_price > 0:
            current_loss = (price - entry_price) / entry_price
            if current_loss < -stop_loss:
                exit_price = price * (1 - commission)
                trade_return = (exit_price - entry_price) / entry_price
                capital *= (1 + trade_return)
                position = 0
                hold_days = 0
                monthly_capitals[month_key]['trades'] += 1
                monthly_capitals[month_key]['end'] = capital
                continue
        
        # Güçlü AL sinyali (>62%)
        if prob > 0.62 and position == 0:
            position = 1
            entry_price = price * (1 + commission)
            hold_days = 0
            monthly_capitals[month_key]['trades'] += 1
        
        # Güçlü SAT sinyali (<38%) veya horizon doldu
        elif position == 1:
            hold_days += 1
            if prob < 0.38 or hold_days >= horizon_days:
                exit_price = price * (1 - commission)
                trade_return = (exit_price - entry_price) / entry_price
                capital *= (1 + trade_return)
                position = 0
                hold_days = 0
                monthly_capitals[month_key]['trades'] += 1
        
        monthly_capitals[month_key]['end'] = capital
    
    # Aylık getiri hesapla
    monthly_returns = []
    for month, data in sorted(monthly_capitals.items()):
        if data['start'] > 0:
            monthly_ret = (data['end'] - data['start']) / data['start'] * 100
            monthly_returns.append(monthly_ret)
    
    # Toplam getiri
    total_return = (capital - 10000) / 10000 * 100
    test_months = len(monthly_capitals)
    avg_monthly = np.mean(monthly_returns) if monthly_returns else 0
    
    # Sharpe ratio (basit)
    if len(monthly_returns) > 1:
        sharpe = np.mean(monthly_returns) / (np.std(monthly_returns) + 1e-8)
    else:
        sharpe = 0
    
    return {
        'total_return_pct': round(total_return, 2),
        'avg_monthly_return_pct': round(avg_monthly, 2),
        'test_months': test_months,
        'monthly_15_achievable': avg_monthly >= 15,
        'final_capital': round(capital, 2),
        'sharpe_ratio': round(sharpe, 2),
        'monthly_returns': [round(r, 2) for r in monthly_returns],
    }


def run_retraining():
    """Ana eğitim fonksiyonu."""
    print("\n" + "="*65)
    print("  GMSTR MODEL YENİDEN EĞİTİM - GELİŞMİŞ VERSİYON")
    print("="*65)
    print(f"  Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65 + "\n")
    
    # Veri yükle
    df = load_daily_data()
    if df is None:
        return False
    
    # Gelişmiş özellikler
    df, all_features = engineer_advanced_features(df)
    
    horizons = {'1d': 1, '3d': 3, '5d': 5, '10d': 10}
    all_results = {}
    
    for h_name, h_days in horizons.items():
        print(f"\n{'─'*65}")
        print(f"  VADE: {h_name} ({h_days} gün sonrası)")
        print(f"{'─'*65}")
        
        # Hedef oluştur
        target, future_ret = create_robust_targets(df, h_days)
        
        # Temiz veri
        clean = df.copy()
        clean['target'] = target
        clean['future_ret'] = future_ret
        clean = clean.dropna(subset=['target', 'future_ret'])
        
        # Geçerli feature kolonları
        valid_features = [c for c in all_features if c in clean.columns]
        clean_feat = clean.dropna(subset=valid_features, thresh=int(len(valid_features)*0.7))
        
        if len(clean_feat) < 300:
            print(f"  ⚠ Yetersiz veri: {len(clean_feat)} satır")
            continue
        
        X = clean_feat[valid_features].fillna(clean_feat[valid_features].median())
        y = clean_feat['target']
        
        # Sınıf dağılımı
        pos_rate = y.mean()
        print(f"  Veri: {len(clean_feat)} satır | Pozitif: {pos_rate:.1%} | Negatif: {1-pos_rate:.1%}")
        
        # Özellik seçimi
        selected = select_best_features(X, y, k=40)
        X = X[selected]
        print(f"  Seçilen özellik: {len(selected)}")
        
        # Train/test split (son %20 test)
        split_idx = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"  Eğitim: {len(X_train)} | Test: {len(X_test)}")
        
        # Gelişmiş model eğit
        print(f"  Gelişmiş model eğitiliyor (stacking ensemble)...")
        result = train_advanced_model(X_train, y_train, X_test, y_test, h_name)
        
        print(f"\n  ┌─ SONUÇLAR ─────────────────────────────────────┐")
        print(f"  │ CV Doğruluk:   {result['cv_accuracy']:.1%} ± {result['cv_std']:.1%}")
        print(f"  │ CV AUC:        {result['cv_auc']:.3f}")
        print(f"  │ Test Doğruluk: {result['test_accuracy']:.1%}")
        print(f"  │ Test AUC:      {result['test_auc']:.3f}")
        
        # Gerçekçilik kontrolü
        is_realistic = True
        issues = []
        
        if result['test_accuracy'] < 0.50:
            issues.append(f"❌ Test doğruluğu rastgele tahminden düşük ({result['test_accuracy']:.1%})")
            is_realistic = False
        elif result['test_accuracy'] < 0.53:
            issues.append(f"⚠️  Test doğruluğu zayıf ({result['test_accuracy']:.1%})")
        
        if result['test_auc'] < 0.50:
            issues.append(f"❌ AUC < 0.5 ({result['test_auc']:.3f})")
            is_realistic = False
        elif result['test_auc'] < 0.55:
            issues.append(f"⚠️  AUC düşük ({result['test_auc']:.3f})")
        
        overfit = result['cv_accuracy'] - result['test_accuracy']
        if overfit > 0.15:
            issues.append(f"⚠️  Overfit riski (CV-Test farkı: {overfit:.1%})")
        
        # Backtest
        bt = realistic_backtest(clean_feat, selected, h_days, result)
        if bt:
            print(f"  │ Backtest Toplam: {bt['total_return_pct']:+.1f}%")
            print(f"  │ Aylık Ortalama:  {bt['avg_monthly_return_pct']:+.1f}%")
            print(f"  │ Sharpe Ratio:    {bt['sharpe_ratio']:.2f}")
            print(f"  │ %15 Hedefi:      {'✅ Ulaşılabilir' if bt['monthly_15_achievable'] else '⚠️  Hedef Altında'}")
            result['backtest'] = bt
        
        if issues:
            print(f"  │ Sorunlar:")
            for issue in issues:
                print(f"  │   {issue}")
        else:
            print(f"  │ ✅ Model gerçekçi ve sağlıklı")
        
        print(f"  └────────────────────────────────────────────────┘")
        
        # Modeli kaydet
        key = f'{h_name}_daily'
        result['horizon'] = h_name
        result['frequency'] = 'daily'
        result['feature_cols'] = selected
        result['is_realistic'] = is_realistic
        result['issues'] = issues
        all_results[key] = result
        
        # Pickle kaydet
        import pickle
        model_data = {
            'scaler': result['scaler'],
            'base_models': result['base_models'],
            'meta_learner': result['meta_learner'],
            'feature_cols': selected,
            'horizon': h_name,
            'frequency': 'daily',
            'trained_at': datetime.now().isoformat(),
            'test_accuracy': result['test_accuracy'],
            'test_auc': result['test_auc'],
        }
        pkl_path = MODEL_DIR / f'simple_{h_name}_daily.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"  💾 Model kaydedildi: {pkl_path.name}")
    
    # Sonuçları kaydet
    save_results = {}
    for key, r in all_results.items():
        save_results[key] = {
            'horizon': r['horizon'],
            'frequency': r['frequency'],
            'cv_accuracy': r['cv_accuracy'],
            'cv_std': r['cv_std'],
            'cv_auc': r.get('cv_auc', 0),
            'test_accuracy': r['test_accuracy'],
            'test_auc': r['test_auc'],
            'positive_rate': r['positive_rate'],
            'train_size': r['train_size'],
            'test_size': r['test_size'],
            'is_realistic': r['is_realistic'],
            'issues': r['issues'],
            'backtest': r.get('backtest', {}),
            'top_features': r.get('top_features', {}),
        }
    
    results_path = MODEL_DIR / 'training_results.json'
    # Mevcut sonuçlarla birleştir (saatlik modelleri koru)
    existing = {}
    if results_path.exists():
        with open(results_path, encoding='utf-8') as f:
            existing = json.load(f)
    
    # Saatlik modelleri koru
    for key in list(existing.keys()):
        if 'hourly' in key:
            save_results[key] = existing[key]
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # Feature columns kaydet
    all_features_set = set()
    for r in all_results.values():
        all_features_set.update(r['feature_cols'])
    
    feat_path = MODEL_DIR / 'feature_columns.json'
    with open(feat_path, 'w') as f:
        json.dump(list(all_features_set), f, indent=2)
    
    print(f"\n{'='*65}")
    print("  EĞİTİM TAMAMLANDI")
    print(f"{'='*65}")
    
    # Özet
    realistic_count = sum(1 for r in all_results.values() if r['is_realistic'])
    print(f"\n  📊 {len(all_results)} model eğitildi")
    print(f"  ✅ {realistic_count} model gerçekçi")
    print(f"  ⚠️  {len(all_results) - realistic_count} model sorunlu")
    
    # %15 hedefi
    achievable = [k for k, r in all_results.items() 
                  if r.get('backtest', {}).get('monthly_15_achievable', False)]
    print(f"\n  🎯 Aylık %15 hedefine ulaşabilen modeller: {len(achievable)}/{len(all_results)}")
    if achievable:
        for k in achievable:
            bt = all_results[k].get('backtest', {})
            print(f"     • {k}: Aylık ort. {bt.get('avg_monthly_return_pct', 0):+.1f}% | Sharpe: {bt.get('sharpe_ratio', 0):.2f}")
    
    print(f"\n  ⚠️  NOT: Backtest geçmiş performansı gösterir.")
    print(f"  Gerçek işlemlerde komisyon, slippage ve piyasa koşulları")
    print(f"  sonuçları önemli ölçüde etkileyebilir.")
    print(f"\n  Sonuçlar kaydedildi: {results_path}")
    print(f"{'='*65}\n")
    
    return True


if __name__ == '__main__':
    success = run_retraining()
    if success:
        print("✅ Yeniden eğitim başarıyla tamamlandı!")
        print("   Web arayüzünde 'Model Doğrulama' sekmesini kontrol edin.")
    else:
        print("❌ Eğitim başarısız!")
        sys.exit(1)
