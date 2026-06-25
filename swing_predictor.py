"""
GMSTR Swing Tahmin Modeli V3 - GELISTIRILMIS
Strateji: Destek/Direnç tepkisi + momentum divergence + hacim onayı
Hedef: %65+ dogruluk
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import logging
from datetime import datetime
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class GMSTRSwingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_cols = []
        self.last_train_time = None
        
    def fetch_data(self, period="2y", interval="1h"):
        ticker = yf.Ticker("GMSTR.IS")
        data = ticker.history(period=period, interval=interval)
        if data.empty or len(data) < 200:
            return None
        return data
    
    def find_pivots(self, highs, lows, order=5):
        """Pivot high/low noktalarını bul"""
        pivot_highs = argrelextrema(highs.values, np.greater, order=order)[0]
        pivot_lows = argrelextrema(lows.values, np.less, order=order)[0]
        return pivot_highs, pivot_lows
    
    def create_features(self, df):
        """Swing odaklı gelismis ozellikler"""
        f = pd.DataFrame(index=df.index)
        o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
        
        # === 1. Fiyat Yapisi ===
        f['returns_1h'] = c.pct_change()
        f['returns_3h'] = c.pct_change(3)
        f['returns_6h'] = c.pct_change(6)
        f['returns_12h'] = c.pct_change(12)
        
        # === 2. EMA Sistemi ===
        emas = {}
        for span in [5, 8, 13, 21, 34]:
            emas[span] = c.ewm(span=span).mean()
            f[f'ema_{span}_dist'] = (c - emas[span]) / emas[span]
        
        f['ema_5_8_cross'] = (emas[5] > emas[8]).astype(int)
        f['ema_8_13_cross'] = (emas[8] > emas[13]).astype(int)
        f['ema_13_21_cross'] = (emas[13] > emas[21]).astype(int)
        f['ema_slope'] = (emas[8] - emas[8].shift(3)) / emas[8].shift(3)
        
        # === 3. RSI Divergence ===
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        f['rsi'] = rsi
        f['rsi_3h_change'] = rsi.diff(3)
        f['rsi_6h_change'] = rsi.diff(6)
        
        # Price-RSI divergence
        price_3h = c.diff(3)
        rsi_3h = rsi.diff(3)
        f['rsi_divergence'] = np.where((price_3h < 0) & (rsi_3h > 0), 1,
                               np.where((price_3h > 0) & (rsi_3h < 0), -1, 0))
        
        # === 4. MACD ===
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        f['macd'] = macd / c
        f['macd_signal'] = signal / c
        f['macd_hist'] = hist / c
        f['macd_hist_slope'] = hist.diff(3)
        f['macd_cross'] = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).astype(int)
        
        # === 5. Bollinger ===
        bb_ma = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        f['bb_position'] = (c - (bb_ma - 2*bb_std)) / (4*bb_std + 1e-10)
        f['bb_squeeze'] = (bb_std / bb_ma).rolling(10).mean() < (bb_std / bb_ma).rolling(50).mean()
        
        # === 6. Stochastic ===
        low_14 = l.rolling(14).min()
        high_14 = h.rolling(14).max()
        f['stoch_k'] = 100 * (c - low_14) / (high_14 - low_14 + 1e-10)
        f['stoch_d'] = f['stoch_k'].rolling(3).mean()
        f['stoch_cross'] = (f['stoch_k'] > f['stoch_d']).astype(int)
        f['stoch_oversold'] = (f['stoch_k'] < 20).astype(int)
        f['stoch_overbought'] = (f['stoch_k'] > 80).astype(int)
        
        # === 7. Pivot Seviyeleri ===
        pivot_highs, pivot_lows = self.find_pivots(h, l, order=5)
        
        # Son pivot high'dan uzaklik
        f['dist_last_pivot_high'] = 999.0
        f['dist_last_pivot_low'] = 999.0
        
        for i in range(len(df)):
            ph_before = [ph for ph in pivot_highs if ph < i]
            pl_before = [pl for pl in pivot_lows if pl < i]
            
            if ph_before:
                last_ph = max(ph_before)
                f.iloc[i, f.columns.get_loc('dist_last_pivot_high')] = (h.iloc[last_ph] - c.iloc[i]) / c.iloc[i]
            
            if pl_before:
                last_pl = max(pl_before)
                f.iloc[i, f.columns.get_loc('dist_last_pivot_low')] = (c.iloc[i] - l.iloc[last_pl]) / c.iloc[i]
        
        # === 8. Hacim Onayı ===
        f['vol_ratio'] = v / v.rolling(20).mean()
        f['vol_trend'] = (v.rolling(5).mean() / v.rolling(20).mean())
        
        # OBV
        obv = pd.Series(index=c.index, dtype=float)
        obv.iloc[0] = v.iloc[0]
        for i in range(1, len(c)):
            if c.iloc[i] > c.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + v.iloc[i]
            elif c.iloc[i] < c.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - v.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        f['obv_slope'] = (obv - obv.shift(6)) / obv.shift(6).replace(0, np.nan)
        
        # === 9. ATR ===
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        f['atr_ratio'] = atr / c
        f['atr_trend'] = (atr.rolling(5).mean() / atr.rolling(20).mean())
        
        # === 10. Mum Deseni ===
        body = abs(c - o)
        range_ = h - l
        f['body_ratio'] = body / (range_ + 1e-10)
        f['bullish_pin'] = ((l == pd.concat([c, o], axis=1).min(axis=1)) & (f['body_ratio'] < 0.3)).astype(int)
        f['bearish_pin'] = ((h == pd.concat([c, o], axis=1).max(axis=1)) & (f['body_ratio'] < 0.3)).astype(int)
        
        # === 11. Volatilite Regimi ===
        vol_5 = c.pct_change().rolling(5).std() * np.sqrt(24)
        vol_20 = c.pct_change().rolling(20).std() * np.sqrt(24)
        f['vol_ratio'] = vol_5 / vol_20.replace(0, np.nan)
        f['vol_expanding'] = (vol_5 > vol_20).astype(int)
        
        # === 12. Trend Gucu ===
        f['adx_proxy'] = abs(ema12 - ema26) / c
        
        # === 13. Destek/Direnç Teması ===
        f['near_resistance'] = (f['dist_last_pivot_high'] < 0.02).astype(int)
        f['near_support'] = (f['dist_last_pivot_low'] < 0.02).astype(int)
        f['mid_range'] = ((f['dist_last_pivot_high'] > 0.02) & (f['dist_last_pivot_low'] > 0.02)).astype(int)
        
        # Temizle
        f = f.replace([np.inf, -np.inf], 0).fillna(0)
        self.feature_cols = list(f.columns)
        return f
    
    def create_smart_labels(self, df, features):
        """
        Akilli etiketleme:
        1. Dirençten dönüş = SELL sinyali
        2. Destekten dönüş = BUY sinyali
        3. Diğer = Yatay
        """
        c = df['Close'].values
        h = df['High'].values
        l = df['Low'].values
        
        labels = np.full(len(df), 2)  # Default yatay
        
        # Direnç/destek noktalarını bul
        pivot_highs, pivot_lows = self.find_pivots(pd.Series(h), pd.Series(l), order=5)
        
        # Her nokta için ileriye bak
        for i in range(20, len(df) - 12):
            current = c[i]
            future_high = np.max(h[i+1:i+13])  # 12 saat sonraki en yüksek
            future_low = np.min(l[i+1:i+13])   # 12 saat sonraki en düşük
            
            # Son pivotlara olan mesafe
            ph_before = [ph for ph in pivot_highs if ph < i and i - ph <= 50]
            pl_before = [pl for pl in pivot_lows if pl < i and i - pl <= 50]
            
            last_ph = max(ph_before) if ph_before else None
            last_pl = max(pl_before) if pl_before else None
            
            resistance_dist = (h[last_ph] - current) / current if last_ph else 999
            support_dist = (current - l[last_pl]) / current if last_pl else 999
            
            # Sadece önemli seviyelerde tahmin yap
            if resistance_dist < 0.015:  # Direnç yakınında
                if future_low < current * 0.985:  # %1.5 düştü
                    labels[i] = 0  # Düşüş
                elif future_high > current * 1.01:  # Kırılım
                    labels[i] = 1
            elif support_dist < 0.015:  # Destek yakınında
                if future_high > current * 1.015:  # %1.5 yükseldi
                    labels[i] = 1  # Yükseliş
                elif future_low < current * 0.99:  # Kırılım
                    labels[i] = 0
        
        return labels.astype(int)
    
    def train(self):
        logger.info("Swing V3 egitimi basliyor...")
        
        data = self.fetch_data("2y", "1h")
        if data is None:
            return False
        
        features = self.create_features(data)
        labels = self.create_smart_labels(data, features)
        
        X = features.values
        y = labels
        
        # Sadece yukarı/aşağı (yatay hariç)
        mask = y != 2
        X = X[mask]
        y = y[mask]
        
        if len(X) < 200:
            logger.error(f"Yetersiz eğitim verisi: {len(X)}")
            return False
        
        logger.info(f"Eğitim verisi: {len(X)} ornek, Sinif dagilimi: {np.bincount(y)}")
        
        # Walk-forward
        window_train = 1500
        window_test = 150
        
        all_preds = []
        all_true = []
        all_proba = []
        
        n_splits = 0
        start = window_train
        
        while start + window_test < len(X) and n_splits < 5:
            X_train = X[start-window_train:start]
            y_train = y[start-window_train:start]
            X_test = X[start:start+window_test]
            y_test = y[start:start+window_test]
            
            if len(np.unique(y_train)) < 2:
                start += window_test
                continue
            
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # PCA
            pca = PCA(n_components=min(25, X_train_s.shape[1]), random_state=42)
            X_train_p = pca.fit_transform(X_train_s)
            X_test_p = pca.transform(X_test_s)
            
            # LightGBM
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            
            model = lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_samples=15,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            
            model.fit(X_train_p, y_train)
            
            pred = model.predict(X_test_p)
            proba = model.predict_proba(X_test_p)[:, 1]
            
            all_preds.extend(pred)
            all_true.extend(y_test)
            all_proba.extend(proba)
            
            start += window_test
            n_splits += 1
        
        if len(all_preds) < 50:
            logger.error(f"Yetersiz test: {len(all_preds)}")
            return False
        
        acc = accuracy_score(all_true, all_preds)
        
        # Guven sinyalleri (0.55 threshold)
        all_proba = np.array(all_proba)
        confident = np.abs(all_proba - 0.5) > 0.05
        conf_acc = accuracy_score(np.array(all_true)[confident], 
                                  (all_proba[confident] > 0.5).astype(int)) if np.sum(confident) > 10 else 0
        
        logger.info(f"Swing V3 - Genel: %{acc*100:.1f}, Guvenli: %{conf_acc*100:.1f} (n={np.sum(confident)}), Toplam: {len(all_preds)}")
        
        if acc < 0.53:
            logger.warning(f"Dusuk dogruluk: %{acc*100:.1f}")
            return False
        
        # Final model
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=min(25, X_s.shape[1]), random_state=42)
        X_p = self.pca.fit_transform(X_s)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.015,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            reg_alpha=0.5,
            reg_lambda=1.0,
            min_child_samples=15,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.model.fit(X_p, y)
        self.last_train_time = datetime.now()
        
        # Feature importance log
        importance = self.model.feature_importances_
        top_idx = np.argsort(importance)[-10:][::-1]
        logger.info("En onemli 10 ozellik:")
        for idx in top_idx:
            logger.info(f"  {self.feature_cols[idx]}: {importance[idx]:.3f}")
        
        return True
    
    def predict(self):
        if self.model is None:
            if not self.train():
                return None
        
        data = self.fetch_data("60d", "1h")
        if data is None:
            return None
        
        features = self.create_features(data)
        latest = features.iloc[-1:].values
        
        latest_s = self.scaler.transform(latest)
        latest_p = self.pca.transform(latest_s)
        
        proba = self.model.predict_proba(latest_p)[0][1]
        pred = 1 if proba > 0.5 else 0
        
        # Guven skoru
        confidence = abs(proba - 0.5) * 2  # 0 -> 0, 0.5 -> 1
        
        current_price = data['Close'].iloc[-1]
        
        # Swing sartlari
        is_near_support = features['near_support'].iloc[-1] == 1
        is_near_resistance = features['near_resistance'].iloc[-1] == 1
        rsi_val = features['rsi'].iloc[-1]
        
        context = ""
        if is_near_support and pred == 1:
            context = "Destek tepkisi"
        elif is_near_resistance and pred == 0:
            context = "Direnc donusu"
        elif rsi_val < 30 and pred == 1:
            context = "Asiri satim donusu"
        elif rsi_val > 70 and pred == 0:
            context = "Asiri alim donusu"
        else:
            context = "Genel momentum"
        
        return {
            'direction': 'YUKSELIS' if pred == 1 else 'DUSUS',
            'confidence': float(confidence),
            'raw_proba': float(proba),
            'current_price': float(current_price),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'context': context,
            'rsi': float(rsi_val),
            'near_support': bool(is_near_support),
            'near_resistance': bool(is_near_resistance)
        }
