#!/usr/bin/env python3
"""
GMSTR Advanced - Daha Güçlü Özellik Mühendisliği ve Kalibrasyon
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json, os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import ta, joblib

HORIZONS = {'1h': 1, '4h': 4, '1d': 8, '1w': 40}
THRESHOLD = 0.0005
MODEL_DIR = '/home/claude/gmstr_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_data(n_days=730, seed=42):
    np.random.seed(seed)
    dates = []
    current = datetime(2023, 1, 2, 10, 0)
    while len(dates) < n_days * 8:
        if current.weekday() < 5 and 10 <= current.hour < 18:
            dates.append(current)
        current += timedelta(hours=1)
    n = len(dates)
    prices = [900.0]
    volumes = []
    regime = 'neutral'
    rc = 0
    params = {
        'bull':{'mu':0.0009,'sigma':0.012},
        'bear':{'mu':-0.0006,'sigma':0.015},
        'neutral':{'mu':0.0003,'sigma':0.008},
        'volatile':{'mu':0.0001,'sigma':0.025},
    }
    for i in range(1, n):
        rc += 1
        if rc > np.random.randint(10, 60):
            regime = np.random.choice(['bull','bear','neutral','volatile'], p=[0.30,0.22,0.35,0.13])
            rc = 0
        hr = dates[i].hour
        iv = 1.0 + 0.35*np.exp(-0.5*((hr-10)/2.5)**2)
        shock = np.random.normal(params[regime]['mu'], params[regime]['sigma']*iv)
        if np.random.random() < 0.004:
            shock += np.random.choice([-1,1]) * np.random.uniform(0.025,0.07)
        prices.append(prices[-1]*np.exp(shock))
        bv = 50000 + 40000*abs(shock)/params[regime]['sigma']
        volumes.append(int(bv*np.random.lognormal(0, 0.45)))
    volumes.append(volumes[-1])
    
    opens, highs, lows = [], [], []
    for p in prices:
        noise = p * 0.0025
        o = p + np.random.uniform(-noise, noise)
        h = p + abs(np.random.normal(0, noise*1.5))
        l = p - abs(np.random.normal(0, noise*1.5))
        opens.append(o); highs.append(max(o,h,p)); lows.append(min(o,l,p))
    return pd.DataFrame({'Open':opens,'High':highs,'Low':lows,'Close':prices,'Volume':volumes},
                        index=pd.DatetimeIndex(dates))

def compute_features(df):
    df = df.copy()
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    
    # RSI multi-period
    for p in [5,7,9,14,21]:
        df[f'rsi_{p}'] = ta.momentum.RSIIndicator(c, p).rsi()
    
    # Stochastic
    st = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df['stoch_k'] = st.stoch()
    df['stoch_d'] = st.stoch_signal()
    df['stoch_kd'] = df['stoch_k'] - df['stoch_d']
    
    # MACD
    for fast, slow, sig in [(12,26,9),(5,13,5),(8,17,9)]:
        macd = ta.trend.MACD(c, slow, fast, sig)
        tag = f'_{fast}'
        df[f'macd{tag}'] = macd.macd()
        df[f'macd_sig{tag}'] = macd.macd_signal()
        df[f'macd_diff{tag}'] = macd.macd_diff()
    
    # EMA/SMA
    for w in [5, 8, 13, 21, 34, 55]:
        df[f'ema{w}'] = ta.trend.EMAIndicator(c, w).ema_indicator()
    for w in [10, 20, 50]:
        df[f'sma{w}'] = ta.trend.SMAIndicator(c, w).sma_indicator()
    
    # Bollinger Bands
    for std in [1.5, 2.0, 2.5]:
        bb = ta.volatility.BollingerBands(c, 20, std)
        tag = str(std).replace('.','')
        df[f'bb_pct{tag}'] = bb.bollinger_pband()
        df[f'bb_w{tag}']   = bb.bollinger_wband()
    
    # ATR + normalized
    df['atr14'] = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    df['atr7']  = ta.volatility.AverageTrueRange(h, l, c, 7).average_true_range()
    df['natr']  = df['atr14'] / c
    
    # ADX
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    df['adx_diff'] = df['adx_pos'] - df['adx_neg']
    
    # CCI
    df['cci20'] = ta.trend.CCIIndicator(h, l, c, 20).cci()
    df['cci14'] = ta.trend.CCIIndicator(h, l, c, 14).cci()
    
    # Williams %R
    df['wr'] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
    
    # OBV + derivatives
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df['obv_ema'] = ta.trend.EMAIndicator(df['obv'], 10).ema_indicator()
    df['obv_diff'] = df['obv'] - df['obv_ema']
    
    # VWAP
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v, 14).volume_weighted_average_price()
    df['vwap_dev'] = (c - df['vwap']) / df['vwap']
    
    # MFI
    df['mfi'] = ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index()
    
    # CMF
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow()
    
    # Returns at multiple lags
    for lag in [1,2,3,4,5,6,8,10,12,16,24,32,40]:
        df[f'ret{lag}'] = c.pct_change(lag)
    
    # Price position relative to EMA
    for w in [5,8,13,21,34,55]:
        df[f'pp_ema{w}'] = (c - df[f'ema{w}']) / df[f'ema{w}']
    
    # EMA crossovers (binary signals)
    df['ema5_8']   = (df['ema5']  > df['ema8']).astype(int)
    df['ema8_13']  = (df['ema8']  > df['ema13']).astype(int)
    df['ema13_21'] = (df['ema13'] > df['ema21']).astype(int)
    df['ema21_34'] = (df['ema21'] > df['ema34']).astype(int)
    df['ema34_55'] = (df['ema34'] > df['ema55']).astype(int)
    df['ema_score'] = (df['ema5_8'] + df['ema8_13'] + df['ema13_21'] + df['ema21_34'] + df['ema34_55'])
    
    # RSI divergence
    for p in [5,7,14]:
        df[f'rsi{p}_mom'] = df[f'rsi_{p}'].diff(4)
    df['price_mom4'] = c.pct_change(4)
    df['div_14'] = df['rsi14_mom'] * np.sign(df['price_mom4'])  # same dir = no div
    
    # Volatility regimes
    df['rv8']  = df['ret1'].rolling(8).std()
    df['rv24'] = df['ret1'].rolling(24).std()
    df['rv40'] = df['ret1'].rolling(40).std()
    df['vol_ratio'] = df['rv8'] / df['rv24'].replace(0, np.nan)
    df['vol_regime'] = (df['rv8'] > df['rv24']).astype(int)  # 1=expanding vol
    
    # Volume patterns
    df['vol_ma20'] = v.rolling(20).mean()
    df['vol_ratio20'] = v / df['vol_ma20'].replace(0, np.nan)
    df['vol_surge'] = (df['vol_ratio20'] > 2).astype(int)
    df['vol_ma_slope'] = df['vol_ma20'].pct_change(5)
    
    # Candle body
    df['body']   = (c - df['Open']) / c
    df['hl_rng'] = (h - l) / c
    df['upper_wick'] = (h - c.clip(lower=df['Open'])) / c
    df['lower_wick'] = (c.clip(upper=df['Open']) - l) / c
    
    # HL position (c in the day's range)
    df['hl_pos'] = (c - l) / (h - l + 1e-8)
    
    # Time features (cyclical)
    df['hr_sin'] = np.sin(2*np.pi*df.index.hour/8)
    df['hr_cos'] = np.cos(2*np.pi*df.index.hour/8)
    df['dw_sin'] = np.sin(2*np.pi*df.index.dayofweek/5)
    df['dw_cos'] = np.cos(2*np.pi*df.index.dayofweek/5)
    
    # Composite momentum score (rules-based + ML synergy)
    df['rsi_z'] = (df['rsi_14'] - df['rsi_14'].rolling(40).mean()) / df['rsi_14'].rolling(40).std()
    df['mom_score'] = (
        np.sign(df['macd_diff_12']) * 0.20 +
        np.sign(df['pp_ema5'])     * 0.15 +
        np.sign(df['pp_ema13'])    * 0.10 +
        (df['rsi_14'] - 50)/50    * 0.15 +
        (df['stoch_k'] - 50)/50   * 0.10 +
        (df['cci20']/200).clip(-1,1) * 0.10 +
        (df['adx_diff']/100).clip(-1,1) * 0.10 +
        np.sign(df['obv_diff'])    * 0.10
    )
    
    # Support/resistance proximity (rolling max/min)
    for w in [10, 20, 40]:
        df[f'dist_high{w}'] = (h.rolling(w).max() - c) / c
        df[f'dist_low{w}']  = (c - l.rolling(w).min()) / c
    
    return df

def make_targets(df):
    for name, bars in HORIZONS.items():
        fr = df['Close'].pct_change(bars).shift(-bars)
        df[f'target_{name}'] = (fr > THRESHOLD).astype(int)
        df[f'fret_{name}'] = fr
    return df

def get_feature_cols(df):
    excl = {'Open','High','Low','Close','Volume','regime'}
    return [c for c in df.columns 
            if c not in excl 
            and not c.startswith('target_') 
            and not c.startswith('fret_')]

def build_xgb():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.65, min_child_weight=5,
        gamma=0.15, reg_alpha=0.2, reg_lambda=1.5,
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, n_jobs=-1)

def build_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        num_leaves=25, subsample=0.75, colsample_bytree=0.65,
        min_child_samples=15, reg_alpha=0.2, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbose=-1)

def build_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=7, min_samples_leaf=8,
        max_features=0.5, random_state=42, n_jobs=-1)

def train_stacked_ensemble(X_tr, y_tr, X_te, y_te):
    """2-level stacking: base models -> meta logistic"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    base_models = [
        ('xgb1', build_xgb()),
        ('lgb1', build_lgb()),
        ('rf1',  build_rf()),
        ('et1',  ExtraTreesClassifier(n_estimators=150, max_depth=7, 
                                       min_samples_leaf=8, max_features=0.5,
                                       random_state=43, n_jobs=-1)),
        ('gb1',  GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                             learning_rate=0.06, subsample=0.75,
                                             random_state=42)),
    ]
    
    # OOF predictions for meta-learner
    tscv = TimeSeriesSplit(n_splits=4)
    oof_preds = np.zeros((len(X_tr_s), len(base_models)))
    te_preds  = np.zeros((len(X_te_s),  len(base_models)))
    
    for j, (name, mdl) in enumerate(base_models):
        oof = np.zeros(len(X_tr_s))
        for train_i, val_i in tscv.split(X_tr_s):
            clone_mdl = type(mdl)(**mdl.get_params())
            clone_mdl.fit(X_tr_s[train_i], y_tr.iloc[train_i])
            oof[val_i] = clone_mdl.predict_proba(X_tr_s[val_i])[:,1]
        oof_preds[:, j] = oof
        
        # Retrain on full train
        mdl.fit(X_tr_s, y_tr)
        te_preds[:, j] = mdl.predict_proba(X_te_s)[:,1]
    
    # Meta-learner
    meta = LogisticRegression(C=0.5, random_state=42)
    meta.fit(oof_preds, y_tr)
    
    final_proba = meta.predict_proba(te_preds)[:,1]
    final_preds = (final_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_te, final_preds)
    auc = roc_auc_score(y_te, final_proba)
    
    # High confidence filter
    hc_mask = (final_proba > 0.62) | (final_proba < 0.38)
    hc_acc  = accuracy_score(y_te[hc_mask], final_preds[hc_mask]) if hc_mask.sum() > 20 else None
    hc_ratio = hc_mask.mean()
    
    return {
        'scaler': scaler,
        'base_models': [m for _, m in base_models],
        'base_names': [n for n, _ in base_models],
        'meta': meta,
        'acc': acc, 'auc': auc,
        'hc_acc': hc_acc, 'hc_ratio': hc_ratio,
        'te_preds': te_preds,
    }

def cv_evaluate(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, aucs = [], []
    for tr_i, val_i in tscv.split(X):
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X.iloc[tr_i])
        X_val = scaler.transform(X.iloc[val_i])
        
        # Quick ensemble
        m1 = build_xgb(); m1.set_params(n_estimators=100); m1.fit(X_tr, y.iloc[tr_i])
        m2 = build_lgb(); m2.set_params(n_estimators=100); m2.fit(X_tr, y.iloc[tr_i])
        
        p1 = m1.predict_proba(X_val)[:,1]
        p2 = m2.predict_proba(X_val)[:,1]
        avg_p = (p1 + p2) / 2
        preds = (avg_p > 0.5).astype(int)
        accs.append(accuracy_score(y.iloc[val_i], preds))
        aucs.append(roc_auc_score(y.iloc[val_i], avg_p))
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)

def run_full_training(df_raw):
    print("Özellikler hesaplanıyor...")
    df = compute_features(df_raw)
    df = make_targets(df)
    feat_cols = get_feature_cols(df)
    
    all_results = {}
    all_models  = {}
    
    for h_name in HORIZONS:
        tgt = f'target_{h_name}'
        clean = df.dropna(subset=feat_cols + [tgt])
        X, y = clean[feat_cols], clean[tgt]
        
        sp = int(len(X)*0.80)
        X_tr, X_te = X.iloc[:sp], X.iloc[sp:]
        y_tr, y_te = y.iloc[:sp], y.iloc[sp:]
        
        print(f"\n[{h_name}] Train={len(X_tr)} Test={len(X_te)} Pos={y_tr.mean():.1%}")
        
        print(f"  CV değerlendirme...")
        cv_acc, cv_std, cv_auc, cv_auc_std = cv_evaluate(X_tr, y_tr, 5)
        
        print(f"  Stacked ensemble eğitiliyor (5 model)...")
        model_data = train_stacked_ensemble(X_tr, y_tr, X_te, y_te)
        
        print(f"  ✓ Test Acc={model_data['acc']:.2%}  AUC={model_data['auc']:.3f}  "
              f"CV={cv_acc:.2%}±{cv_std:.2%}")
        if model_data['hc_acc']:
            print(f"    Yüksek güven: {model_data['hc_acc']:.2%} ({model_data['hc_ratio']:.0%} veri)")
        
        all_results[h_name] = {
            'cv_acc': round(cv_acc, 4), 'cv_std': round(cv_std, 4),
            'cv_auc': round(cv_auc, 4),
            'test_acc': round(model_data['acc'], 4),
            'test_auc': round(model_data['auc'], 4),
            'hc_acc': round(model_data['hc_acc'], 4) if model_data['hc_acc'] else None,
            'hc_ratio': round(model_data['hc_ratio'], 4),
        }
        all_models[h_name] = {
            'scaler': model_data['scaler'],
            'base_models': model_data['base_models'],
            'meta': model_data['meta'],
            'features': feat_cols,
        }
        
        # Save
        joblib.dump(all_models[h_name], f"{MODEL_DIR}/stack_{h_name}.pkl")
    
    with open(f"{MODEL_DIR}/results_adv.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(f"{MODEL_DIR}/feat_cols.json", 'w') as f:
        json.dump(feat_cols, f)
    
    return all_results, all_models, feat_cols

def predict_current(df_raw, all_models, feat_cols):
    df = compute_features(df_raw.tail(300))
    clean = df.dropna(subset=feat_cols)
    if len(clean) == 0: return {}
    
    last = clean.iloc[[-1]]
    X_last = last[feat_cols]
    cur_p = df_raw['Close'].iloc[-1]
    
    preds = {}
    for h_name, bars in HORIZONS.items():
        md = all_models[h_name]
        X_s = md['scaler'].transform(X_last)
        
        base_p = np.array([m.predict_proba(X_s)[0,1] for m in md['base_models']])
        meta_p = md['meta'].predict_proba(base_p.reshape(1,-1))[0,1]
        
        vol = df['ret1'].std() * np.sqrt(bars)
        move = vol * (meta_p - 0.5) * 2
        
        preds[h_name] = {
            'prob_up': round(float(meta_p), 4),
            'direction': 'YUKARI ↑' if meta_p > 0.5 else 'AŞAĞI ↓',
            'confidence': round(float(max(meta_p, 1-meta_p)), 4),
            'cur_price': round(cur_p, 2),
            'pred_price': round(cur_p * (1+move), 2),
            'exp_chg_pct': round(move*100, 2),
            'base_probs': [round(float(p),3) for p in base_p],
        }
    return preds

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        from gmstr_model import load_real_data
        df_raw = load_real_data(sys.argv[1])
        print(f"Gerçek veri: {len(df_raw)} satır")
    else:
        print("Sentetik GMSTR verisi (2 yıl saatlik)...")
        df_raw = generate_data(730)
    
    results, models, feat_cols = run_full_training(df_raw)
    
    print("\n" + "="*65)
    print("STACKED ENSEMBLE PERFORMANS ÖZETİ")
    print("="*65)
    print(f"{'Ufuk':<5} {'CV Acc':<20} {'Test Acc':<12} {'AUC':<8} {'YüksekGüven'}")
    print("-"*65)
    for h, r in results.items():
        hc = f"{r['hc_acc']:.0%}@{r['hc_ratio']:.0%}" if r['hc_acc'] else "N/A"
        print(f"{h:<5} {r['cv_acc']:.2%}±{r['cv_std']:.2%}        "
              f"{r['test_acc']:.2%}       {r['test_auc']:.3f}   {hc}")
    
    preds = predict_current(df_raw, models, feat_cols)
    print("\nGÜNCEL TAHMİNLER:")
    for h, p in preds.items():
        s = "🟢" if "YUKARI" in p['direction'] else "🔴"
        print(f"  {s} {h}: {p['direction']}  güven={p['confidence']:.0%}  "
              f"→{p['pred_price']:.2f} TRY ({p['exp_chg_pct']:+.2f}%)")
    
    out = {'results': results, 'predictions': preds}
    with open('/home/claude/gmstr_advanced_out.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSonuçlar: /home/claude/gmstr_advanced_out.json")
