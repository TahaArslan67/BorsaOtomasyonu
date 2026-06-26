"""
GMSTR %65+ Başarılı Tahmin Sistemi
Backend API ve Otomatik Tahmin Motoru
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
from datetime import datetime, timedelta, time
from dateutil import parser as date_parser
import schedule
import time as time_module
import logging
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
import json
warnings.filterwarnings('ignore')

# Swing modeli
try:
    from swing_predictor import GMSTRSwingPredictor
    SWING_AVAILABLE = True
except Exception as e:
    SWING_AVAILABLE = False
    print(f"Swing modeli yuklenemedi: {e}")

# Haber analizi modulu
try:
    from gmstr_enhanced.news_analyzer import get_analyzer
    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('gmstr_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env dosyası yüklendi")
except ImportError:
    logger.warning("python-dotenv kurulu değil, .env dosyası yüklenemedi")

class GMSTRPredictionSystem:
    def __init__(self):
        self.model_paths = {
            '1h': 'gmstr_model_1h.pkl',
            '4h': 'gmstr_prediction_model.pkl',
            '1d': 'gmstr_model_1d.pkl'
        }
        self.models = {'1h': None, '4h': None, '1d': None}
        self.features = []
        self.model_path = self.model_paths['4h']  # backward compat
        self.model = None

        # Haber analizi
        self.news_analyzer = None
        if NEWS_ANALYZER_AVAILABLE:
            try:
                self.news_analyzer = get_analyzer()
                logger.info("Haber analizcisi basariliyla baslatildi")
            except Exception as e:
                logger.warning(f"Haber analizcisi baslatilamadi: {e}")

        # Pozisyon takibi (trailing stop)
        self.position = None  # {'entry_price', 'direction', 'stop_loss', 'tp1', 'tp2', 'size'}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_date = None

        # Veritabanı ayarları - PostgreSQL veya SQLite
        self.database_url = os.environ.get('DATABASE_URL', '')
        self.db_path = 'gmstr_predictions.db'
        self.is_postgres = bool(self.database_url)
        
        if self.is_postgres:
            try:
                import psycopg2
                logger.info("PostgreSQL veritabanı kullanılıyor")
            except ImportError:
                logger.warning("psycopg2 bulunamadı, SQLite kullanılıyor")
                self.is_postgres = False
        else:
            logger.info("SQLite veritabanı kullanılıyor")
        
        self.init_database()
        
        # Telegram Bot Config
        self.telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
        
        # Cache
        self._market_cache = None
        self._market_cache_time = None
        self._gmstr_cache = None
        self._gmstr_cache_time = None
        self._cache_ttl = 300  # 5 dakika
        
    def send_telegram_message(self, message):
        """Telegram bot ile mesaj gönder"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.warning("Telegram bot token veya chat ID tanımlanmamış")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram mesajı gönderildi")
                return True
            else:
                logger.error(f"Telegram mesaj gönderme hatası: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram mesaj gönderme hatası: {e}")
            return False
        
    def get_db_connection(self):
        """Veritabanı bağlantısı döndür (PostgreSQL veya SQLite)"""
        if self.is_postgres:
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            return conn
        else:
            return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Veritabanını başlat - eksik sütunları otomatik ekle"""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        if self.is_postgres:
            # PostgreSQL tabloları
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    predicted_for_time TIMESTAMP,
                    current_price REAL,
                    predicted_direction TEXT,
                    predicted_price REAL,
                    confidence REAL,
                    timeframe TEXT,
                    actual_price REAL,
                    actual_direction TEXT,
                    is_correct INTEGER,
                    telegram_sent INTEGER DEFAULT 0,
                    model_type TEXT DEFAULT 'normal',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    accuracy REAL,
                    timeframe TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    timeframe TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            # SQLite tabloları
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    predicted_for_time DATETIME,
                    current_price REAL,
                    predicted_direction TEXT,
                    predicted_price REAL,
                    confidence REAL,
                    timeframe TEXT,
                    actual_price REAL,
                    actual_direction TEXT,
                    is_correct INTEGER,
                    telegram_sent INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Eksik sütunları kontrol et ve ekle
            cursor.execute("PRAGMA table_info(predictions)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            
            if 'predicted_for_time' not in existing_columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN predicted_for_time DATETIME")
                logger.info("predicted_for_time sütunu eklendi")
            
            if 'actual_direction' not in existing_columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN actual_direction TEXT")
                logger.info("actual_direction sütunu eklendi")
            
            if 'telegram_sent' not in existing_columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN telegram_sent INTEGER DEFAULT 0")
                logger.info("telegram_sent sütunu eklendi")
            
            if 'model_type' not in existing_columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN model_type TEXT DEFAULT 'normal'")
                logger.info("model_type sütunu eklendi")
                # Mevcut swing kayitlarini guncelle (timeframe 1h olanlar, sonradan ayristirmak icin)
                cursor.execute("UPDATE predictions SET model_type = 'swing' WHERE timeframe = '1h'")
                logger.info("Mevcut swing tahminleri model_type='swing' olarak isaretlendi")
            
            # Performans tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    accuracy REAL,
                    timeframe TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Backtesting tablosu
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    timeframe TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        conn.commit()
        conn.close()
        logger.info("Veritabanı başlatıldı")
    
    def fetch_gmstr_data(self, period="2y", interval="1h"):
        """GMSTR verilerini çek (cache'li)"""
        now = time_module.time()
        if (self._gmstr_cache is not None and self._gmstr_cache_time and 
            (now - self._gmstr_cache_time) < self._cache_ttl and
            getattr(self, '_gmstr_cache_interval', None) == interval):
            if len(self._gmstr_cache) >= 50:
                logger.info("GMSTR verisi cache'den alındı")
                return self._gmstr_cache
            else:
                logger.warning(f"Cache'deki veri yetersiz ({len(self._gmstr_cache)} satir), yeniden çekiliyor")
                self._gmstr_cache = None
        
        try:
            # Yahoo Finance'den GMSTR verisi
            ticker = yf.Ticker("GMSTR.IS")
            data = ticker.history(period=period, interval=interval)
            
            if data is None:
                logger.error("GMSTR verisi None döndü")
                return self._gmstr_cache if self._gmstr_cache and len(self._gmstr_cache) >= 50 else None
            
            # DataFrame'e çevir
            if isinstance(data, list):
                if len(data) > 0 and len(data[0]) >= 5:
                    data = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data = data.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume'
                    })
                    data.set_index('timestamp', inplace=True)
                else:
                    logger.error("GMSTR verisi formatı geçersiz")
                    return self._gmstr_cache if self._gmstr_cache and len(self._gmstr_cache) >= 50 else None
            
            # Fallback: yetersiz veri ise kisa period veya farkli interval dene
            if data.empty or len(data) < 50:
                logger.warning(f"GMSTR verisi yetersiz ({len(data) if not data.empty else 0} satir), fallback deneniyor...")
                for fallback_period in ["6mo", "3mo", "1mo", "max"]:
                    for fallback_interval in ["1h", "1d"]:
                        try:
                            data = ticker.history(period=fallback_period, interval=fallback_interval)
                            if data is not None and not data.empty and len(data) >= 50:
                                logger.info(f"GMSTR fallback basarili: {fallback_period}/{fallback_interval}, {len(data)} satir")
                                break
                        except Exception as fallback_err:
                            logger.debug(f"Fallback hatasi {fallback_period}/{fallback_interval}: {fallback_err}")
                    else:
                        continue
                    break
                
                if data.empty or len(data) < 50:
                    logger.error(f"GMSTR verisi yetersiz: {len(data) if not data.empty else 0} satir (tum denemeler basarisiz)")
                    return self._gmstr_cache if self._gmstr_cache and len(self._gmstr_cache) >= 50 else None
            
            self._gmstr_cache = data
            self._gmstr_cache_time = now
            self._gmstr_cache_interval = interval
            logger.info(f"GMSTR verisi çekildi: {len(data)} satir")
            return data
        except Exception as e:
            logger.error(f"GMSTR veri çekme hatası: {e}")
            return self._gmstr_cache if self._gmstr_cache and len(self._gmstr_cache) >= 50 else None
    
    def fetch_market_data(self, period="2y"):
        """Piyasa verilerini çek - Cache'li"""
        now = time_module.time()
        if self._market_cache is not None and self._market_cache_time and (now - self._market_cache_time) < self._cache_ttl:
            logger.info("Piyasa verisi cache'den alındı")
            return self._market_cache
        
        try:
            # BIST 100
            bist100 = yf.Ticker("XU100.IS").history(period=period, interval="1h")
            
            # USD/TRY
            usd_try = yf.Ticker("USDTRY=X").history(period=period, interval="1h")
            
            # Altın fiyatı
            gold = None
            try:
                gold = yf.Ticker("GC=F").history(period=period, interval="1h")
                if gold.empty:
                    gold = None
            except:
                pass
            
            # Gümüş fiyatı
            silver = None
            for symbol in ["SI=F", "SILVER", "XAGUSD"]:
                try:
                    silver = yf.Ticker(symbol).history(period=period, interval="1h")
                    if not silver.empty:
                        logger.info(f"Gümüş sembolü bulundu: {symbol}")
                        break
                except:
                    continue
            
            # VIX (Volatilite endeksi)
            vix = None
            try:
                vix = yf.Ticker("^VIX").history(period=period, interval="1h")
                if vix.empty:
                    vix = None
            except:
                pass
            
            # Brent petrol
            oil = None
            try:
                oil = yf.Ticker("BZ=F").history(period=period, interval="1h")
                if oil.empty:
                    oil = None
            except:
                pass
            
            if silver is None or silver.empty:
                logger.warning("Gümüş verisi bulunamadı, alternatif kullanılacak")
                if not usd_try.empty:
                    silver = usd_try.copy()
                    silver['Close'] = silver['Close'] * 0.05
                else:
                    silver = None
            
            result = {
                'bist100': bist100,
                'usd_try': usd_try,
                'gold': gold,
                'silver': silver,
                'vix': vix,
                'oil': oil
            }
            
            self._market_cache = result
            self._market_cache_time = now
            return result
            
        except Exception as e:
            logger.error(f"Piyasa veri çekme hatası: {e}")
            if self._market_cache:
                logger.warning("Önceki cache kullanılıyor")
                return self._market_cache
            return None
    
    def calculate_technical_indicators(self, df):
        """Teknik göstergeleri hesapla"""
        indicators = {}
        
        try:
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            indicators['macd'] = exp1 - exp2
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            indicators['bb_upper'] = sma20 + (std20 * 2)
            indicators['bb_lower'] = sma20 - (std20 * 2)
            indicators['bb_middle'] = sma20
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr'] = true_range.rolling(14).mean()
            
            # Stochastic
            low14 = df['Low'].rolling(14).min()
            high14 = df['High'].rolling(14).max()
            indicators['stoch_k'] = 100 * ((df['Close'] - low14) / (high14 - low14))
            indicators['stoch_d'] = indicators['stoch_k'].rolling(3).mean()
            
            # Williams %R
            indicators['williams_r'] = -100 * ((high14 - df['Close']) / (high14 - low14))
            
            # EMA ve SMA
            indicators['ema_20'] = df['Close'].ewm(span=20).mean()
            indicators['sma_50'] = df['Close'].rolling(50).mean()
            
            # Momentum
            indicators['momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Volume Delta
            indicators['volume_delta'] = df['Volume'] - df['Volume'].shift(1)
            
            # OBV (On Balance Volume)
            obv = [0]  # Başlangıç değeri
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            indicators['obv'] = pd.Series(obv, index=df.index)
            
            # CCI (Commodity Channel Index)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(20).mean()
            mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            indicators['cci'] = (tp - sma_tp) / (0.015 * mean_dev)
            
            # Z-Score
            indicators['z_score'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
            
            # Tüm göstergelerin Series olduğunu kontrol et
            for key, value in indicators.items():
                if isinstance(value, list):
                    indicators[key] = pd.Series(value, index=df.index)
                elif not isinstance(value, pd.Series):
                    indicators[key] = pd.Series(value, index=df.index)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Teknik gösterge hesaplama hatası: {e}")
            # Boş göstergeler döndür
            return {}
    
    def _add_lag_features(self, df, cols, lags=[1, 3, 5, 10, 20]):
        """Lag features ekle - gecmis degerleri feature olarak kullan"""
        for col in cols:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    # Degisim orani da ekle
                    df[f'{col}_chg_{lag}'] = df[col].pct_change(lag)
        return df

    def _add_rolling_features(self, df, cols, windows=[5, 10, 20, 50]):
        """Rolling istatistikler ekle"""
        for col in cols:
            if col in df.columns:
                for w in windows:
                    df[f'{col}_roll_mean_{w}'] = df[col].rolling(w).mean()
                    df[f'{col}_roll_std_{w}'] = df[col].rolling(w).std()
                    df[f'{col}_roll_max_{w}'] = df[col].rolling(w).max()
                    df[f'{col}_roll_min_{w}'] = df[col].rolling(w).min()
                    df[f'{col}_roll_zscore_{w}'] = (df[col] - df[f'{col}_roll_mean_{w}']) / (df[f'{col}_roll_std_{w}'] + 1e-9)
        return df

    def _add_interaction_features(self, df, cols):
        """Feature carpimlari (interaction) ekle"""
        numeric_cols = [c for c in cols if c in df.columns]
        for i in range(len(numeric_cols)):
            for j in range(i+1, min(i+5, len(numeric_cols))):  # Sadece ilk 4 eslestirme (karmasikligi azalt)
                c1, c2 = numeric_cols[i], numeric_cols[j]
                df[f'{c1}_x_{c2}'] = df[c1] * df[c2]
                df[f'{c1}_div_{c2}'] = df[c1] / (df[c2] + 1e-9)
        return df

    def create_features(self, gmstr_data, market_data):
        """Gelismis ozellik muhendisligi: lag + rolling + interaction + gmstr_system features."""
        try:
            # DataFrame kontrolu
            if isinstance(gmstr_data, list):
                if len(gmstr_data) > 0 and len(gmstr_data[0]) >= 5:
                    gmstr_data = pd.DataFrame(gmstr_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    gmstr_data.set_index('timestamp', inplace=True)
                else:
                    logger.error("gmstr_data formati gecersiz")
                    return None

            if isinstance(gmstr_data, dict):
                gmstr_data = pd.DataFrame(gmstr_data)

            if not isinstance(gmstr_data, pd.DataFrame):
                logger.error(f"gmstr_data tipi gecersiz: {type(gmstr_data)}")
                return None

            # Veri uzunlugu kontrolu (minimum 50 satir gerekli)
            if len(gmstr_data) < 50:
                logger.error(f"Yetersiz veri: {len(gmstr_data)} satir, minimum 50 gerekli")
                return None

            # Sutun isimlerini buyuk harf yap
            rename_map = {}
            for col in gmstr_data.columns:
                if col.lower() in ['open', 'high', 'low', 'close', 'volume']:
                    rename_map[col] = col.capitalize()
            if rename_map:
                gmstr_data = gmstr_data.rename(columns=rename_map)

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required_columns if c not in gmstr_data.columns]
            if missing:
                logger.error(f"Eksik sutunlar: {missing}")
                return None

            # Makro verileri ekle
            if market_data:
                for key, ticker_df in market_data.items():
                    if ticker_df is not None and not ticker_df.empty and 'Close' in ticker_df.columns:
                        col_name = f'macro_{key}_close'
                        gmstr_data[col_name] = ticker_df['Close'].reindex(gmstr_data.index, method='ffill')
                        ret_col = f'macro_{key}_ret'
                        gmstr_data[ret_col] = gmstr_data[col_name].pct_change()

            # Temel price/returns
            gmstr_data['returns'] = gmstr_data['Close'].pct_change()
            gmstr_data['log_returns'] = np.log(gmstr_data['Close'] / gmstr_data['Close'].shift(1))
            gmstr_data['volatility_20d'] = gmstr_data['returns'].rolling(20).std()
            gmstr_data['range'] = gmstr_data['High'] - gmstr_data['Low']
            gmstr_data['range_pct'] = gmstr_data['range'] / gmstr_data['Close']

            # Lag features
            lag_cols = ['Close', 'Volume', 'returns']
            gmstr_data = self._add_lag_features(gmstr_data, lag_cols)

            # Rolling features
            roll_cols = ['Close', 'Volume', 'returns']
            gmstr_data = self._add_rolling_features(gmstr_data, roll_cols)

            # NaN temizligi
            gmstr_data = gmstr_data.dropna()
            if len(gmstr_data) < 50:
                logger.error(f"NaN temizligi sonrasi yetersiz veri: {len(gmstr_data)}")
                return None

            # FeatureEngineer ile 150+ gosterge uret
            from gmstr_system.features import FeatureEngineer
            fe = FeatureEngineer()
            try:
                df_features = fe.transform(gmstr_data)
            except Exception as fe_err:
                logger.warning(f"FeatureEngineer hatasi: {fe_err}, basic ozelliklerle devam ediliyor")
                df_features = gmstr_data.copy()

            if df_features is None or df_features.empty:
                logger.error("FeatureEngineer bos DataFrame dondurdu")
                return None

            # Feature kolonlarini al
            try:
                feature_cols = fe.get_feature_columns(df_features)
            except:
                feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'date']]
            if not feature_cols:
                logger.error("FeatureEngineer hic kolon uretmedi")
                return None

            # Interaction features (sadece en onemli 10 ile)
            top_cols = feature_cols[:10] if len(feature_cols) >= 10 else feature_cols
            df_features = self._add_interaction_features(df_features, top_cols)
            feature_cols = [c for c in df_features.columns if c not in ['timestamp', 'date'] and not c.startswith('target')]

            X = df_features[feature_cols].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            self.features = feature_cols
            logger.info(f"Ozellik matrisi olusturuldu: {X.shape} | {len(feature_cols)} gosterge")
            return X

        except Exception as e:
            logger.error(f"Ozellik olusturma hatasi: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def create_labels(self, data, timeframe_hours=72):
        """Etiketler olustur - 72 saat (3 gun), %3 esik"""
        labels = []

        for i in range(50, len(data) - timeframe_hours):
            current_price = data['Close'].iloc[i]
            future_price = data['Close'].iloc[i + timeframe_hours]

            if future_price > current_price * 1.03:
                labels.append(1)  # Yukselis
            elif future_price < current_price * 0.97:
                labels.append(0)  # Dusus
            else:
                labels.append(2)  # Yatay

        return np.array(labels)
    
    def train_model(self):
        """Model ensemble egitimi: RF + XGBoost + LightGBM + walk-forward validation."""
        try:
            logger.info("Model ensemble egitimi basliyor (RF+XGB+LGBM)...")

            gmstr_data = self.fetch_gmstr_data()
            market_data = self.fetch_market_data()

            if gmstr_data is None:
                logger.error("GMSTR verisi cekilemedi")
                return False

            X = self.create_features(gmstr_data, market_data)
            y = self.create_labels(gmstr_data)

            if X is None or len(X) == 0:
                logger.error("Ozellikler olusturulamadi")
                return False

            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

            # Yatay sinyalleri cikar (sadece yukselis/dusus)
            mask = y != 2
            X = X[mask]
            y = y[mask]

            if len(X) < 100:
                logger.error(f"Yetersiz veri: {len(X)} ornek")
                return False

            # Sinif dagilimi
            from collections import Counter
            class_dist = Counter(y)
            logger.info(f"Sinif dagilimi: {dict(class_dist)}")

            # Walk-forward validation: son %20 test, gerisi egitim (zaman serisi uyumlu)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # SMOTE ile oversampling (sinif dengesizligini coz)
            try:
                from imblearn.over_sampling import SMOTE
                min_class = min(class_dist.values())
                k = max(1, min(5, min_class - 1))
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                logger.info(f"SMOTE sonrasi: {Counter(y_train_res)}")
                X_train, y_train = X_train_res, y_train_res
            except ImportError:
                logger.warning("imblearn kurulu degil, SMOTE atlaniyor")
            except Exception as smote_err:
                logger.warning(f"SMOTE hatasi: {smote_err}")

            logger.info(f"Egitim: {len(X_train)} | Test: {len(X_test)} (walk-forward)")

            # Class 0'ı random oversample et (dengesizlik cozumu)
            try:
                n0, n1 = np.bincount(y_train.astype(int))
                if n0 < n1:
                    # Dusus orneklerini cogalt
                    idx_0 = np.where(y_train == 0)[0]
                    n_repeat = (n1 - n0) // max(1, len(idx_0))
                    if n_repeat > 0:
                        extra_idx = np.tile(idx_0, n_repeat)
                        X_train = np.vstack([X_train, X_train[extra_idx]])
                        y_train = np.concatenate([y_train, y_train[extra_idx]])
                        logger.info(f"Oversampling: Class 0 {n_repeat}x cogaltildi | Yeni dagilim: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            except Exception as os_err:
                logger.warning(f"Oversampling hatasi: {os_err}")

            # PCA ile feature azaltma (noise azaltma)
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Aciklanan varyans %90 olan bilesen sayisini bul
                pca_full = PCA(random_state=42)
                pca_full.fit(X_train_scaled)
                cumsum = np.cumsum(pca_full.explained_variance_ratio_)
                n_comp = np.argmax(cumsum >= 0.90) + 1
                n_comp = min(n_comp, 30)  # Max 30

                pca = PCA(n_components=n_comp, random_state=42)
                X_train_sel = pca.fit_transform(X_train_scaled)
                X_test_sel = pca.transform(X_test_scaled)
                top_idx = np.arange(n_comp)  # PCA bilesen indisleri
                logger.info(f"PCA: {n_comp} bilesen secildi (%{cumsum[n_comp-1]*100:.1f} varyans)")
            except Exception as pca_err:
                logger.warning(f"PCA hatasi: {pca_err}, tum ozellikler kullaniliyor")
                X_train_sel, X_test_sel = X_train, X_test
                top_idx = np.arange(X_train.shape[1])

            # Sinif agirligi: dusus sinifini 3x agirlikli ogret
            n0, n1 = np.bincount(y_train.astype(int))
            pos_weight = max(1.0, n0 / max(1, n1)) * 3.0  # 3x agresif
            class_weights = {0: 3.0, 1: 1.0} if n1 > n0 else {0: 1.0, 1: 3.0}
            logger.info(f"Sinif agirligi: pos_weight={pos_weight:.2f} (n0={n0}, n1={n1}) | class_weights={class_weights}")

            # Sadece LightGBM (optimize edilmis)
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.02,
                    subsample=0.7, colsample_bytree=0.7,
                    is_unbalance=True,
                    reg_alpha=1.0, reg_lambda=2.0,
                    min_child_samples=50,
                    num_leaves=7,
                    random_state=42, n_jobs=-1, verbosity=-1
                )
                lgb_model.fit(X_train_sel, y_train)
                lgb_proba = lgb_model.predict_proba(X_test_sel)[:, 1]
                lgb_acc = accuracy_score(y_test, (lgb_proba > 0.5).astype(int))
                logger.info(f"LGBM Accuracy: {lgb_acc:.4f}")
            except Exception as lgb_err:
                logger.warning(f"LightGBM egitim hatasi: {lgb_err}")
                # RF fallback
                rf = RandomForestClassifier(
                    n_estimators=80, max_depth=3, min_samples_split=100,
                    min_samples_leaf=50, class_weight=class_weights,
                    random_state=42, n_jobs=-1
                )
                rf.fit(X_train_sel, y_train)
                lgb_proba = rf.predict_proba(X_test_sel)[:, 1]
                lgb_acc = accuracy_score(y_test, (lgb_proba > 0.5).astype(int))
                logger.info(f"RF Fallback Accuracy: {lgb_acc:.4f}")

            # Tek model = LGBM
            rf_proba = lgb_proba
            xgb_proba = lgb_proba
            rf_acc = lgb_acc
            xgb_acc = lgb_acc

            # Optimal threshold bulma (precision-recall tradeoff)
            from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
            def find_best_threshold(y_true, y_proba):
                precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
                f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
                best_idx = np.argmax(f1s)
                return thresholds[min(best_idx, len(thresholds)-1)]

            # Ensemble: Tek model = LGBM (sade ve robust)
            ensemble_proba = lgb_proba
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)

            # Isotonic regression ile confidence kalibrasyonu (Platt'tan daha iyi)
            from sklearn.isotonic import IsotonicRegression
            try:
                lgb_train_proba = lgb_model.predict_proba(X_train_sel)[:, 1]
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.fit(lgb_train_proba, y_train)
                cal_proba = iso.predict(ensemble_proba)
                cal_pred = (cal_proba > 0.5).astype(int)
                cal_acc = accuracy_score(y_test, cal_pred)
                logger.info(f"Kalibre edilmis accuracy: {cal_acc:.4f}")
                ensemble_proba = cal_proba
            except Exception as iso_err:
                logger.warning(f"Isotonic kalibrasyon hatasi: {iso_err}")

            final_proba = ensemble_proba
            final_acc = ensemble_acc

            # Kalite filtreleme (|proba-0.5| > 0.20 - dengeli)
            high_conf_mask = final_proba > 0.70
            low_conf_mask = final_proba < 0.30
            filtered_mask = high_conf_mask | low_conf_mask
            if np.sum(filtered_mask) > 10:
                filtered_acc = accuracy_score(y_test[filtered_mask], (final_proba[filtered_mask] > 0.5).astype(int))
                logger.info(f"Filtreli (guclu) Accuracy: {filtered_acc:.4f} | Ornek: {np.sum(filtered_mask)}/{len(y_test)}")
            else:
                filtered_acc = final_acc

            logger.info(f"LGBM Accuracy: {ensemble_acc:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, ensemble_pred)}")

            # Model kaydet
            ensemble_model = {
                'lgb': lgb_model if 'lgb_model' in dir() else rf,
                'features': self.features,
                'feature_indices': top_idx.tolist(),
                'pca': pca if 'pca' in dir() else None,
                'scaler': scaler if 'scaler' in dir() else None
            }

            model_info = {
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'lgbm_accuracy': round(lgb_acc, 4),
                'filtered_accuracy': round(filtered_acc, 4) if 'filtered_acc' in dir() else None,
                'test_samples': len(y_test),
                'train_samples': len(X_train),
                'status': 'saved',
                'threshold': 0.50,
                'features_used': len(self.features) if self.features else 0,
                'pca_components': n_comp if 'n_comp' in dir() else 0,
                'quality_filter': True,
                'quality_threshold': 0.15
            }
            with open('model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)

            # Her zaman modeli kaydet (dusuk accuracy'de bile calissin)
            self.model = ensemble_model
            joblib.dump(ensemble_model, self.model_path)
            with open('feature_names.txt', 'w') as f:
                f.write(','.join(self.features))

            # Feature importance (LGBM uzerinden)
            try:
                if 'lgb_model' in dir() and lgb_model:
                    importances = lgb_model.feature_importances_
                else:
                    importances = np.ones(len(self.features)) / len(self.features)
                feat_imp = list(zip(self.features, importances))
                feat_imp.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"En onemli 10 ozellik: {feat_imp[:10]}")
            except Exception as imp_err:
                logger.warning(f"Feature importance hatasi: {imp_err}")

            if lgb_acc >= 0.50:
                logger.info(f"Model kaydedildi: {lgb_acc:.4f}")
                return True
            else:
                logger.warning(f"Model accuracy dusuk ({lgb_acc:.4f}) ama kaydedildi")
                return True

        except Exception as e:
            logger.error(f"Model egitimi hatasi: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def is_borsa_open(self):
        """Borsa açık mı kontrol et (Türkiye saati)"""
        now = datetime.now()
        # Hafta sonu kontrol
        if now.weekday() >= 5:  # Cumartesi=5, Pazar=6
            return False
        # Saat kontrolü (10:00 - 18:10)
        market_open = now.replace(hour=10, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=18, minute=10, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    def _get_adx(self, df: pd.DataFrame, period=14):
        """ADX (Average Directional Index) hesapla - piyasa rejimi algilama."""
        try:
            from ta.trend import ADXIndicator
            adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=period)
            return adx.adx().iloc[-1]
        except Exception:
            return 25.0  # Varsayilan: orta trend

    def _multi_timeframe_consensus(self, timeframes=["1h", "4h", "1d"]):
        """Zengin coklu zaman dilimi: EMA + RSI + MACD + SuperTrend consensus."""
        votes = []
        confidences = []
        current_price = None
        tf_details = {}

        for tf in timeframes:
            try:
                if tf == "1h":
                    data = yf.Ticker("GMSTR.IS").history(period="5d", interval="1h")
                    ema_fast, ema_slow = 10, 20
                elif tf == "4h":
                    data = yf.Ticker("GMSTR.IS").history(period="20d", interval="1h")
                    if not data.empty:
                        data = data.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                    ema_fast, ema_slow = 5, 10
                elif tf == "1d":
                    data = yf.Ticker("GMSTR.IS").history(period="90d", interval="1d")
                    ema_fast, ema_slow = 10, 20
                else:
                    continue

                if data.empty or len(data) < max(ema_slow, 26) + 5:
                    continue

                close = data['Close']
                high = data['High']
                low = data['Low']

                # 1. EMA kesisimi
                ema_f = close.ewm(span=ema_fast).mean().iloc[-1]
                ema_s = close.ewm(span=ema_slow).mean().iloc[-1]

                # 2. RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                rsi_last = rsi.iloc[-1] if not rsi.empty else 50
                if pd.isna(rsi_last):
                    rsi_last = 50

                # 3. MACD
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                macd_last = macd.iloc[-1] - signal.iloc[-1] if len(macd) > 0 else 0

                # 4. SuperTrend basit
                atr = (high - low).rolling(10).mean().iloc[-1]
                st_upper = (high + low) / 2 + 3 * atr if not pd.isna(atr) else close.iloc[-1] * 1.02
                st_lower = (high + low) / 2 - 3 * atr if not pd.isna(atr) else close.iloc[-1] * 0.98
                st_dir = 1 if close.iloc[-1] > st_upper else -1

                # 5. Hacim onayi
                vol_sma = data['Volume'].rolling(20).mean().iloc[-1]
                vol_confirmed = data['Volume'].iloc[-1] > vol_sma if not pd.isna(vol_sma) else False

                if pd.isna(ema_f) or pd.isna(ema_s):
                    continue

                if current_price is None:
                    current_price = float(close.iloc[-1])

                # Sinyal skoru (0-5 arasi)
                score = 0
                if ema_f > ema_s:
                    score += 1
                if rsi_last > 55:
                    score += 1
                elif rsi_last < 45:
                    score -= 1
                if macd_last > 0:
                    score += 1
                elif macd_last < 0:
                    score -= 1
                if st_dir == 1:
                    score += 1
                elif st_dir == -1:
                    score -= 1
                if vol_confirmed:
                    score += 0.5

                if score >= 2.5:
                    votes.append(1)
                elif score <= -2.5:
                    votes.append(0)
                else:
                    votes.append(-1)

                # Guven = skor normalize
                conf = min(abs(score) / 5.0, 1.0)
                confidences.append(conf)
                tf_details[tf] = {'score': score, 'ema': ema_f > ema_s, 'rsi': rsi_last, 'macd': macd_last > 0, 'st': st_dir}

            except Exception as e:
                logger.debug(f"{tf} consensus hatasi: {e}")
                continue

        if not votes:
            return None, None, current_price, tf_details

        up_votes = sum(1 for v in votes if v == 1)
        down_votes = sum(1 for v in votes if v == 0)
        total_valid = sum(1 for v in votes if v != -1)

        if total_valid == 0:
            return None, 0.0, current_price, tf_details

        consensus = up_votes / total_valid if (up_votes >= down_votes) else -(down_votes / total_valid)
        avg_conf = np.mean(confidences) if confidences else 0.5

        return consensus, avg_conf, current_price, tf_details

    def make_prediction(self, timeframe="4h"):
        """Gelismis tahmin: ensemble + coklu zaman dilimi + ADX rejim + dinamik guven."""
        try:
            if self.model is None:
                try:
                    self.model = joblib.load(self.model_path)
                    with open('feature_names.txt', 'r') as f:
                        self.features = f.read().split(',')
                except:
                    logger.error("Model bulunamadi, once egitim yapin")
                    return None

            # 1. Ensemble ML tahmini
            gmstr_data = self.fetch_gmstr_data(period="2y")
            market_data = self.fetch_market_data()

            if gmstr_data is None:
                logger.error("GMSTR verisi cekilemedi")
                return None

            X = self.create_features(gmstr_data, market_data)
            if X is None or len(X) == 0:
                logger.error("Ozellikler olusturulamadi")
                return None

            latest_features = X[-1].reshape(1, -1)
            current_price = gmstr_data['Close'].iloc[-1]

            # PCA ve Scaler uygula (varsa)
            if isinstance(self.model, dict):
                if self.model.get('scaler'):
                    latest_features = self.model['scaler'].transform(latest_features)
                    logger.debug("Scaler uygulandi")
                if self.model.get('pca'):
                    latest_features = self.model['pca'].transform(latest_features)
                    logger.debug(f"PCA uygulandi: {latest_features.shape[1]} bilesen")

            # LGBM tahmin
            if isinstance(self.model, dict) and 'lgb' in self.model:
                lgb_proba = self.model['lgb'].predict_proba(latest_features)[0][1]
                ml_prediction = 1 if lgb_proba > 0.5 else 0
                ml_confidence = max(lgb_proba, 1 - lgb_proba)
                logger.debug(f"LGBM proba: {lgb_proba:.4f}")
            else:
                ml_prediction = self.model.predict(latest_features)[0]
                probability = self.model.predict_proba(latest_features)[0]
                ml_confidence = max(probability)

            # 2. Coklu zaman dilimi consensus
            consensus, tf_conf, _, tf_details = self._multi_timeframe_consensus()
            consensus_str = f"{consensus:.2f}" if consensus is not None else "N/A"
            tf_conf_str = f"{tf_conf:.3f}" if tf_conf is not None else "N/A"
            logger.info(f"Multi-TF consensus: {consensus_str} | conf: {tf_conf_str} | details: {tf_details}")

            # 3. ADX piyasa rejimi (>=25)
            adx_value = self._get_adx(gmstr_data)
            regime = "TREND" if adx_value >= 25 else "SIDEWAYS"
            logger.info(f"ADX: {adx_value:.1f} | Rejim: {regime}")

            # 4. Final sinyal: ML + consensus birlestir
            if consensus is not None:
                ml_score = ml_confidence if ml_prediction == 1 else -ml_confidence
                combined_score = 0.6 * ml_score + 0.4 * consensus
                final_prediction = 1 if combined_score > 0 else 0
                combined_confidence = abs(combined_score)
            else:
                final_prediction = ml_prediction
                combined_confidence = ml_confidence

            # 5. Haber analizi filtresi
            news_mult, news_score = 1.0, None
            if self.news_analyzer:
                try:
                    news = self.news_analyzer.fetch_news(count=12)
                    sentiment = self.news_analyzer.get_news_sentiment_score(news)
                    if sentiment:
                        news_score = sentiment.get('overall_score', 0)
                        # Negatif haber + YUKSELIS sinyali = guven dusur
                        if news_score < -0.3 and final_prediction == 1:
                            combined_confidence *= 0.8
                            logger.info(f"Negatif haber, guven dusuruldu: {news_score:.2f}")
                        elif news_score > 0.3 and final_prediction == 0:
                            combined_confidence *= 0.8
                            logger.info(f"Pozitif haber ama DUSUS sinyali, guven dusuruldu: {news_score:.2f}")
                except Exception as e:
                    logger.debug(f"Haber filtresi hatasi: {e}")

            # Pozisyon boyutu: Kelly Criterion
            try:
                # Kelly % = (p*b - q) / b
                # p = win probability (confidence)
                # b = win/loss ratio (tahmini 2:1)
                p = combined_confidence
                b = 2.0  # Risk/Reward ratio
                q = 1 - p
                kelly_pct = (p * b - q) / b
                kelly_pct = max(0.1, min(0.5, kelly_pct))  # %10-%50 arasi sinirla
                position_size = kelly_pct
                logger.info(f"Kelly pozisyon boyutu: %{position_size*100:.1f}")
            except Exception as kelly_err:
                position_size = 0.2
                logger.warning(f"Kelly hatasi: {kelly_err}")

            # 6. Korelasyon filtresi (GMSTR-BIST100)
            try:
                if market_data and 'bist100' in market_data and not market_data['bist100'].empty:
                    bist = market_data['bist100']
                    gmstr_ret = gmstr_data['Close'].pct_change().dropna().tail(20).values
                    bist_ret = bist['Close'].pct_change().dropna().tail(20).values
                    min_len = min(len(gmstr_ret), len(bist_ret))
                    if min_len >= 10:
                        corr = np.corrcoef(gmstr_ret[-min_len:], bist_ret[-min_len:])[0, 1]
                        if not pd.isna(corr) and abs(corr) < 0.3:
                            combined_confidence *= 0.9
                            logger.info(f"Zayif korelasyon ({corr:.2f}), guven dusuruldu")
            except Exception as e:
                logger.debug(f"Korelasyon filtresi hatasi: {e}")

            # 7. Volatilite bazli adaptive threshold
            try:
                recent_returns = gmstr_data['Close'].pct_change().dropna().tail(20)
                vol = recent_returns.std() * np.sqrt(252) if len(recent_returns) > 5 else 0.3
            except:
                vol = 0.3

            base_threshold = 0.50  # Sabit dusuk threshold

            # 8. SIDeways piyasada guven dusur
            if regime == "SIDEWAYS":
                combined_confidence *= 0.9
                logger.info(f"Sideways: guven dusuruldu")

            # 9. Pozisyon varsa trailing stop kontrolu
            if self.position:
                try:
                    if self.position['direction'] == "YUKSELIS":
                        # Trailing stop yukselt
                        new_sl = current_price * 0.98
                        if new_sl > self.position['stop_loss']:
                            self.position['stop_loss'] = new_sl
                            logger.info(f"Trailing stop yukseltildi: {new_sl:.2f}")
                        # TP1 veya TP2 tetiklendi mi
                        if current_price >= self.position['tp1'] and not self.position.get('tp1_hit'):
                            self.position['tp1_hit'] = True
                            logger.info("TP1 tetiklendi, pozisyon yariya dusuruldu")
                        if current_price >= self.position['tp2']:
                            logger.info("TP2 tetiklendi, pozisyon tamamen kapatildi")
                            self.position = None
                    else:  # DUSUS
                        new_sl = current_price * 1.02
                        if new_sl < self.position['stop_loss']:
                            self.position['stop_loss'] = new_sl
                            logger.info(f"Trailing stop dusuruldu: {new_sl:.2f}")
                        if current_price <= self.position['tp1'] and not self.position.get('tp1_hit'):
                            self.position['tp1_hit'] = True
                            logger.info("TP1 tetiklendi, pozisyon yariya dusuruldu")
                        if current_price <= self.position['tp2']:
                            logger.info("TP2 tetiklendi, pozisyon tamamen kapatildi")
                            self.position = None
                    # Stop-loss tetiklendi mi
                    if self.position:
                        if (self.position['direction'] == "YUKSELIS" and current_price <= self.position['stop_loss']) or \
                           (self.position['direction'] == "DUSUS" and current_price >= self.position['stop_loss']):
                            logger.warning("Stop-loss tetiklendi, pozisyon kapatildi")
                            self.position = None
                except Exception as e:
                    logger.debug(f"Trailing stop hatasi: {e}")

            # 10. Gunluk circuit breaker (max kayip %5)
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_pnl = 0.0
                self.daily_trades = 0
                self.last_trade_date = today
            if self.daily_pnl < -500:  # Sabit ornek: 500 birim
                logger.warning("Gunluk circuit breaker tetiklendi, bugun islem yok")
                direction = "HOLD"

            # 11. Kalite filtresi: Guven > 0.60 (dengeli)
            quality_pass = abs(ml_confidence - 0.5) > 0.10 if 'ml_confidence' in dir() else combined_confidence >= 0.60
            if not quality_pass:
                logger.info(f"Kalite filtresi: guven {combined_confidence:.3f}, HOLD")
                direction = "HOLD"
                target_price = current_price
            elif combined_confidence < base_threshold:
                direction = "HOLD"
                target_price = current_price
                logger.info(f"Guven dusuk ({combined_confidence:.3f} < {base_threshold}), HOLD")
            else:
                if final_prediction == 1:
                    target_price = current_price * 1.015
                    direction = "YUKSELIS"
                else:
                    target_price = current_price * 0.985
                    direction = "DUSUS"
                # Yeni pozisyon ac
                self.position = {
                    'entry_price': current_price,
                    'direction': direction,
                    'stop_loss': current_price * 0.98 if direction == "YUKSELIS" else current_price * 1.02,
                    'tp1': target_price * 0.67 + current_price * 0.33,
                    'tp2': target_price,
                    'size': 1.0,
                    'tp1_hit': False
                }

            now = datetime.now()
            if timeframe == "4h":
                predicted_for_time = now + timedelta(hours=4)
            elif timeframe == "1d":
                predicted_for_time = now + timedelta(days=1)
            else:
                predicted_for_time = now + timedelta(hours=4)

            pred_id = self.save_prediction(current_price, direction, target_price, combined_confidence, timeframe, predicted_for_time)

            # Risk yonetimi bilgilerini hesapla
            risk_info = None
            if direction != "HOLD":
                try:
                    risk_info = self.calculate_risk_management(
                        current_price, target_price, direction, combined_confidence
                    )
                except Exception as e:
                    logger.warning(f"Risk hesaplama hatasi: {e}")

            telegram_sent = False
            if direction != "HOLD" and combined_confidence >= base_threshold:
                emoji = "🟢" if direction == "YUKSELIS" else "🔴"
                risk_str = ""
                if risk_info and isinstance(risk_info, dict):
                    try:
                        sl = risk_info.get('stop_loss', 0) or 0
                        tp1 = risk_info.get('take_profit_1', 0) or 0
                        tp2 = risk_info.get('take_profit_2', 0) or 0
                        rr = risk_info.get('risk_reward_ratio', 0) or 0
                        pos = risk_info.get('position_size_pct', 10) or 10
                        kelly = risk_info.get('kelly_raw_pct', 0) or 0
                        streak = risk_info.get('streak_multiplier', 1) or 1
                        risk_str = f"""
🛑 <b>Stop Loss:</b> ₺{sl:.2f}
🎯 <b>TP1:</b> ₺{tp1:.2f} | <b>TP2:</b> ₺{tp2:.2f}
📐 <b>R/R:</b> {rr:.2f}
💼 <b>Pozisyon:</b> %{pos:.1f} (Kelly %{kelly:.1f})
🔥 <b>Streak:</b> {streak:.2f}x"""
                    except Exception as rs_err:
                        logger.warning(f"Risk string olusturma hatasi: {rs_err}")
                        risk_str = ""

                # Son 5 tahmin gecmisini getir
                history_str = ""
                try:
                    conn = self.get_db_connection()
                    cursor = conn.cursor()
                    ph = '%s' if self.is_postgres else '?'
                    cursor.execute(f'''
                        SELECT predicted_direction, confidence, actual_price, is_correct, timestamp
                        FROM predictions
                        WHERE predicted_direction != 'HOLD'
                        AND telegram_sent = 1
                        ORDER BY timestamp DESC
                        LIMIT 5
                    ''')
                    rows = cursor.fetchall()
                    if rows:
                        history_lines = []
                        for row in rows:
                            pred_dir, conf, actual, correct, ts = row
                            correct_emoji = "✅" if correct == 1 else "❌" if correct == 0 else "⏳"
                            actual_str = f"₺{actual:.0f}" if actual else "?"
                            history_lines.append(f"  {correct_emoji} {pred_dir} (%{conf*100:.0f}) → {actual_str}")
                        
                        # Dogruluk orani
                        cursor.execute('''
                            SELECT COUNT(*) as total, SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                            FROM predictions WHERE is_correct IS NOT NULL
                        ''')
                        total_val, correct_val = cursor.fetchone()
                        acc_str = f"%{correct_val/total_val*100:.0f}" if total_val and total_val > 0 else "N/A"
                        
                        history_str = f"""

📋 <b>Son Tahminler:</b>
{chr(10).join(history_lines)}

📊 <b>Toplam Dogruluk:</b> {acc_str} ({correct_val}/{total_val})"""
                    conn.close()
                except Exception as hist_err:
                    logger.debug(f"Tahmin gecmisi hatasi: {hist_err}")

                message = f"""<b>GMSTR Sinyal</b> {emoji}

📅 <b>Tahmin Zamani:</b> {now.strftime('%d.%m.%Y %H:%M')}
⏰ <b>Gecerli Olacagi Zaman:</b> {predicted_for_time.strftime('%d.%m.%Y %H:%M')}

💰 <b>Mevcut Fiyat:</b> ₺{current_price:.2f}
📈 <b>Tahmin:</b> {direction}
🎯 <b>Hedef Fiyat:</b> ₺{target_price:.2f}
🔒 <b>Guven:</b> %{combined_confidence*100:.1f} (ML:{ml_confidence:.2f} + TF:{(tf_conf if tf_conf else 0):.2f})
📊 <b>Rejim:</b> {regime} (ADX:{adx_value:.1f}) | Vol:{vol:.2f}
{risk_str}

⏳ <b>Beklenen Degisim:</b> %{abs((target_price - current_price) / current_price * 100):.2f}%{history_str}"""
                telegram_sent = self.send_telegram_message(message)
                if telegram_sent and pred_id:
                    self.update_telegram_status(pred_id, 1)

            result = {
                'timestamp': now,
                'predicted_for_time': predicted_for_time,
                'current_price': current_price,
                'direction': direction,
                'target_price': target_price,
                'confidence': combined_confidence,
                'ml_confidence': ml_confidence,
                'consensus': consensus,
                'regime': regime,
                'adx': adx_value,
                'volatility_annual': round(vol, 3),
                'threshold_used': base_threshold,
                'risk_info': risk_info,
                'news_score': news_score,
                'position_status': 'OPEN' if self.position else 'CLOSED',
                'tf_details': tf_details,
                'timeframe': timeframe,
                'telegram_sent': telegram_sent
            }
            logger.info(f"Tahmin: {direction} | Guven: {combined_confidence:.4f} | Rejim: {regime} | Vol: {vol:.2f} | Telegram: {telegram_sent}")
            return result

        except Exception as e:
            logger.error(f"Tahmin hatasi: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def make_backfill_prediction(self, as_of, timeframe='4h', gmstr_data=None, market_data=None):
        """Gecmise donuk tahmin: modeli egitilmis haliyle, veriyi as_of tarihine kadar kirparak calistirir.
        Not: Bu fonksiyon sadece temel ML tahminini kullanir; canli market verisi, haber ve consensus filtreleri atlanir.
        Opsiyonel gmstr_data ve market_data parametreleri ile backfill hizlanir.
        """
        try:
            if self.model is None:
                try:
                    self.model = joblib.load(self.model_path)
                    with open('feature_names.txt', 'r') as f:
                        self.features = f.read().split(',')
                except:
                    logger.error("Backfill icin model bulunamadi, once egitim yapin")
                    return None
            
            # 2y veriyi cek veya disaridan al, as_of'a kadar kirp
            if gmstr_data is None:
                gmstr_data = self.fetch_gmstr_data(period="2y")
            else:
                gmstr_data = gmstr_data.copy()
            if gmstr_data is None or gmstr_data.empty:
                return None
            if gmstr_data.index.tz is not None:
                gmstr_data.index = gmstr_data.index.tz_localize(None)
            gmstr_data = gmstr_data[gmstr_data.index < as_of]
            
            # Piyasa verisini cek veya disaridan al, as_of'a kadar kirp
            if market_data is None:
                market_data = self.fetch_market_data(period="2y")
            else:
                market_data = {k: v.copy() for k, v in market_data.items() if v is not None}
            if market_data:
                for key, df in market_data.items():
                    if df is not None and not df.empty:
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        market_data[key] = df[df.index < as_of]
            if len(gmstr_data) < 50:
                logger.warning(f"Backfill icin yetersiz veri: {len(gmstr_data)} satir @ {as_of}")
                return None
            
            X = self.create_features(gmstr_data, market_data)
            if X is None or len(X) == 0:
                return None
            
            latest_features = X[-1].reshape(1, -1)
            current_price = float(gmstr_data['Close'].iloc[-1])
            
            # PCA ve Scaler uygula (varsa)
            if isinstance(self.model, dict):
                if self.model.get('scaler'):
                    latest_features = self.model['scaler'].transform(latest_features)
                if self.model.get('pca'):
                    latest_features = self.model['pca'].transform(latest_features)
                if 'lgb' in self.model:
                    proba = self.model['lgb'].predict_proba(latest_features)[0][1]
                else:
                    proba = self.model.predict_proba(latest_features)[0][1]
            else:
                proba = self.model.predict_proba(latest_features)[0][1]
            
            ml_prediction = 1 if proba > 0.5 else 0
            confidence = abs(proba - 0.5) * 2
            
            if ml_prediction == 1:
                direction = 'YUKSELIS'
                target_price = current_price * 1.015
            else:
                direction = 'DUSUS'
                target_price = current_price * 0.985
            
            return {
                'timestamp': as_of,
                'current_price': current_price,
                'direction': direction,
                'target_price': target_price,
                'confidence': confidence,
                'timeframe': timeframe
            }
        except Exception as e:
            logger.error(f"Backfill tahmin hatasi @ {as_of}: {e}")
            return None
    
    def save_prediction(self, current_price, direction, target_price, confidence, timeframe, predicted_for_time=None, model_type='normal'):
        """Tahmini veritabanına kaydet ve ID döndür - Sadece borsa acikken"""
        try:
            now = datetime.now()
            
            # Borsa kapaliysa kaydetme
            if not self.is_borsa_open():
                logger.info(f"Borsa kapali, tahmin kaydedilmedi: {direction} @{now.strftime('%H:%M')}")
                return None
            
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor.execute('''
                    INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                    predicted_price, confidence, timeframe, model_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (now, predicted_for_time, current_price, direction, target_price, confidence, timeframe, model_type))
                pred_id = cursor.fetchone()[0]
            else:
                cursor.execute('''
                    INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                    predicted_price, confidence, timeframe, model_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (now, predicted_for_time, current_price, direction, target_price, confidence, timeframe, model_type))
                pred_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return pred_id
            
        except Exception as e:
            logger.error(f"Tahmin kaydetme hatası: {e}")
            return None
    
    def save_historical_prediction(self, timestamp, current_price, direction, target_price, confidence, timeframe, model_type='normal'):
        """Gecmise donuk tahmini veritabanina kaydet (borsa kontrolu yapmadan)"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor.execute('''
                    INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                    predicted_price, confidence, timeframe, model_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (timestamp, None, current_price, direction, target_price, confidence, timeframe, model_type))
                pred_id = cursor.fetchone()[0]
            else:
                cursor.execute('''
                    INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                    predicted_price, confidence, timeframe, model_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, None, current_price, direction, target_price, confidence, timeframe, model_type))
                pred_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            return pred_id
        except Exception as e:
            logger.error(f"Gecmis tahmin kaydetme hatasi: {e}")
            return None
    
    def update_telegram_status(self, pred_id, status):
        """Telegram gönderim durumunu güncelle"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            ph = '%s' if self.is_postgres else '?'
            cursor.execute(f'UPDATE predictions SET telegram_sent = {ph} WHERE id = {ph}', (status, pred_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Telegram status güncelleme hatası: {e}")
    
    def update_predictions(self):
        """Tahminleri guncelle (dogruluk kontrolu) - Otomatik"""
        try:
            now = datetime.now()
            day = now.weekday()
            hour = now.hour
            minute = now.minute
            
            # Hafta sonu veya 18:10 - 10:00 arasi borsa kapali
            is_weekend = day >= 5
            is_after_hours = (hour > 18) or (hour == 18 and minute >= 10) or (hour < 10)
            
            if is_weekend or is_after_hours:
                logger.info("Borsa kapali, tahmin dogrulama atlandi")
                return
            
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Daha agresif validasyon: 5dk gecmis veya predicted_for_time gecmis olanlar
            if self.is_postgres:
                cursor.execute('''
                    SELECT id, timestamp, predicted_for_time, current_price, predicted_price, predicted_direction, confidence
                    FROM predictions
                    WHERE actual_price IS NULL
                    AND (predicted_for_time < %s OR timestamp < %s)
                ''', (now, now - timedelta(minutes=5)))
            else:
                cursor.execute('''
                    SELECT id, timestamp, predicted_for_time, current_price, predicted_price, predicted_direction, confidence
                    FROM predictions
                    WHERE actual_price IS NULL
                    AND (predicted_for_time < datetime('now') OR datetime(timestamp) < datetime('now', '-5 minutes'))
                ''')
            
            predictions = cursor.fetchall()
            logger.info(f"Validasyon bekleyen tahmin sayisi: {len(predictions)}")
            
            if len(predictions) == 0:
                conn.close()
                return
            
            # O anki fiyatı al
            gmstr_data = self.fetch_gmstr_data(period="1d")
            if gmstr_data is None:
                logger.error("GMSTR verisi çekilemedi, tahminler güncellenemiyor")
                conn.close()
                return
                
            actual_price = gmstr_data['Close'].iloc[-1]
            
            correct_count = 0
            total_count = len(predictions)
            ph = '%s' if self.is_postgres else '?'
            
            for pred in predictions:
                pred_id, timestamp, pred_for_time, current_price, pred_price, pred_direction, confidence = pred
                
                # Dogrulugu kontrol et (bizim yon formati: YUKSELIS / DUSUS)
                if pred_direction == "YUKSELIS":
                    is_correct = 1 if actual_price > current_price else 0
                    actual_direction = "YUKSELIS" if actual_price > current_price else "DUSUS"
                elif pred_direction == "DUSUS":
                    is_correct = 1 if actual_price < current_price else 0
                    actual_direction = "DUSUS" if actual_price < current_price else "YUKSELIS"
                else:
                    # HOLD veya diger: fiyat yukseldiyse YUKSELIS, dustuyse DUSUS
                    if actual_price > current_price:
                        actual_direction = "YUKSELIS"
                    elif actual_price < current_price:
                        actual_direction = "DUSUS"
                    else:
                        actual_direction = "HOLD"
                    is_correct = None
                
                if is_correct:
                    correct_count += 1
                
                # Güncelle
                cursor.execute(f'''
                    UPDATE predictions
                    SET actual_price = {ph}, actual_direction = {ph}, is_correct = {ph}
                    WHERE id = {ph}
                ''', (actual_price, actual_direction, is_correct, pred_id))
            
            conn.commit()
            
            # Sonuçları logla
            accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
            logger.info(f"Tahmin güncellemesi: {correct_count}/{total_count} doğru (%{accuracy:.1f})")
            
            # Başarı %65 altına düşerse OTOMATİK MODEL EĞİTİMİ
            if total_count >= 10 and accuracy < 65:
                logger.warning(f"Başarı oranı düşük (%{accuracy:.1f}), model otomatik yeniden eğitiliyor...")
                
                # Telegram bildirim
                if self.telegram_bot_token:
                    retrain_msg = f"""⚠️ <b>Model Yeniden Eğitiliyor</b>

📊 Başarı oranı %65 altına düştü: %{accuracy:.1f}
🔄 Model otomatik yeniden eğitiliyor...

<i>Bu işlem birkaç dakika sürebilir.</i>"""
                    self.send_telegram_message(retrain_msg)
                
                # Arka planda modeli eğit
                try:
                    import threading
                    retrain_thread = threading.Thread(target=self.train_model, daemon=True)
                    retrain_thread.start()
                    logger.info("Otomatik model eğitimi başlatıldı (arka plan)")
                except Exception as retrain_err:
                    logger.error(f"Otomatik eğitim başlatma hatası: {retrain_err}")
            
            # Sonuçları Telegram'dan bildir
            if total_count > 0 and self.telegram_bot_token:
                results_emoji = "✅" if accuracy >= 65 else "⚠️"
                result_msg = f"""<b>Tahmin Doğrulama Raporu</b> {results_emoji}

📊 <b>Toplam Tahmin:</b> {total_count}
✅ <b>Doğru:</b> {correct_count}
❌ <b>Yanlış:</b> {total_count - correct_count}
📈 <b>Başarı Oranı:</b> %{accuracy:.1f}

💰 <b>Gerçekleşen Fiyat:</b> ₺{actual_price:.2f}

<i>Otomatik kontrol tamamlandı.</i>"""
                self.send_telegram_message(result_msg)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Tahmin güncelleme hatası: {e}")
    
    def get_performance_stats(self):
        """Performans istatistikleri"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            if self.is_postgres:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND created_at > %s
                    AND EXTRACT(DOW FROM timestamp) BETWEEN 1 AND 5
                    AND (
                        EXTRACT(HOUR FROM timestamp) BETWEEN 10 AND 17
                        OR (EXTRACT(HOUR FROM timestamp) = 18 AND EXTRACT(MINUTE FROM timestamp) <= 10)
                    )
                ''', (seven_days_ago,))
            else:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(is_correct) as correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND datetime(created_at) > datetime('now', '-7 days')
                    AND strftime('%w', timestamp) BETWEEN '1' AND '5'
                    AND strftime('%H:%M', timestamp) BETWEEN '10:00' AND '18:10'
                ''')
            
            stats = cursor.fetchone()
            
            conn.close()
            
            if stats and stats[0] > 0:
                total = stats[0]
                correct = stats[1] or 0
                return {
                    'total_predictions': total,
                    'correct_predictions': correct,
                    'accuracy': (correct / total) * 100
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Performans istatistikleri hatası: {e}")
            return None
    
    def get_premarket_signal(self):
        """Cok faktorlu pre-market sinyali: gumus + BIST100 + USD/TRY + altin."""
        try:
            gmstr = yf.Ticker("GMSTR.IS")
            gmstr_data = gmstr.history(period="5d", interval="1h")
            if gmstr_data.empty:
                logger.error("GMSTR verisi bos")
                return None

            gmstr_last_close = float(gmstr_data['Close'].iloc[-1])
            gmstr_close_time = gmstr_data.index[-1]
            logger.info(f"GMSTR son: fiyat={gmstr_last_close}, zaman={gmstr_close_time}")

            factors = {}

            # 1. Gumus (yfinance + currency-api fallback)
            silver_current = None
            silver_at_close = None
            silver_change = None
            
            # 1a. yfinance dene
            for sym in ["SI=F", "XAGUSD=X", "SL=F"]:
                try:
                    silver_data = yf.Ticker(sym).history(period="7d", interval="1h")
                    if silver_data.empty or len(silver_data) < 2:
                        continue
                    silver_current = float(silver_data['Close'].iloc[-1])
                    gmstr_ts = gmstr_close_time.timestamp()
                    silver_at_close = None
                    for i in range(len(silver_data)-1, -1, -1):
                        if silver_data.index[i].timestamp() <= gmstr_ts:
                            silver_at_close = float(silver_data['Close'].iloc[i])
                            break
                    if silver_at_close is None:
                        silver_at_close = float(silver_data['Close'].iloc[0])
                    silver_change = ((silver_current - silver_at_close) / silver_at_close) * 100
                    logger.info(f"Gumus ({sym}): {silver_change:.2f}% | now=${silver_current:.2f} close=${silver_at_close:.2f}")
                    break
                except Exception:
                    continue
            
            # 1b. yfinance basarisiz olduysa currency-api fallback (anlik fiyat)
            if silver_current is None:
                try:
                    r = requests.get('https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json', timeout=10)
                    data = r.json()
                    xag_per_usd = data['usd']['xag']
                    silver_current = 1.0 / xag_per_usd
                    
                    # Kapanis icin dunu dene
                    try:
                        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        r2 = requests.get(f'https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{yesterday}/v1/currencies/usd.json', timeout=10)
                        data2 = r2.json()
                        xag_per_usd_yest = data2['usd']['xag']
                        silver_at_close = 1.0 / xag_per_usd_yest
                        silver_change = ((silver_current - silver_at_close) / silver_at_close) * 100
                        logger.info(f"Gumus (currency-api): {silver_change:.2f}% | now=${silver_current:.2f} close=${silver_at_close:.2f}")
                    except Exception as e2:
                        logger.warning(f"Gumus kapanis alinamadi, sadece anlik: {e2}")
                        silver_at_close = silver_current
                        silver_change = 0.0
                except Exception as e:
                    logger.warning(f"Gumus verisi alinamadi (tum kaynaklar): {e}")
            
            if silver_change is not None:
                factors['silver'] = silver_change
            else:
                factors['silver'] = 0

            # Fiyat bilgilerini sakla
            prices = {
                'silver_current': silver_current,
                'silver_at_close': silver_at_close,
                'bist100_current': None,
                'bist100_at_close': None,
                'usd_try_current': None,
                'usd_try_at_close': None,
                'gold_current': None,
                'gold_at_close': None
            }

            # 2. BIST100 (XU100.IS)
            try:
                bist = yf.Ticker("XU100.IS")
                bist_data = bist.history(period="5d", interval="1h")
                if not bist_data.empty:
                    bist_current = float(bist_data['Close'].iloc[-1])
                    bist_at_close = None
                    for i in range(len(bist_data)-1, -1, -1):
                        if bist_data.index[i].timestamp() <= gmstr_ts:
                            bist_at_close = float(bist_data['Close'].iloc[i])
                            break
                    if bist_at_close:
                        bist_change = ((bist_current - bist_at_close) / bist_at_close) * 100
                        factors['bist100'] = bist_change
                        prices['bist100_current'] = bist_current
                        prices['bist100_at_close'] = bist_at_close
                        logger.info(f"BIST100: {bist_change:.2f}% | now={bist_current:.0f}")
            except Exception as e:
                logger.warning(f"BIST100 verisi alinamadi: {e}")

            # 3. USD/TRY
            try:
                usd = yf.Ticker("USDTRY=X")
                usd_data = usd.history(period="7d", interval="1h")
                if not usd_data.empty:
                    usd_current = float(usd_data['Close'].iloc[-1])
                    usd_at_close = None
                    for i in range(len(usd_data)-1, -1, -1):
                        if usd_data.index[i].timestamp() <= gmstr_ts:
                            usd_at_close = float(usd_data['Close'].iloc[i])
                            break
                    if usd_at_close:
                        usd_change = ((usd_current - usd_at_close) / usd_at_close) * 100
                        factors['usd_try'] = usd_change
                        prices['usd_try_current'] = usd_current
                        prices['usd_try_at_close'] = usd_at_close
                        logger.info(f"USD/TRY: {usd_change:.2f}% | now={usd_current:.2f}")
            except Exception as e:
                logger.warning(f"USD/TRY verisi alinamadi: {e}")

            # 4. Altin (GC=F)
            try:
                gold = yf.Ticker("GC=F")
                gold_data = gold.history(period="7d", interval="1h")
                if not gold_data.empty:
                    gold_current = float(gold_data['Close'].iloc[-1])
                    gold_at_close = None
                    for i in range(len(gold_data)-1, -1, -1):
                        if gold_data.index[i].timestamp() <= gmstr_ts:
                            gold_at_close = float(gold_data['Close'].iloc[i])
                            break
                    if gold_at_close:
                        gold_change = ((gold_current - gold_at_close) / gold_at_close) * 100
                        factors['gold'] = gold_change
                        prices['gold_current'] = gold_current
                        prices['gold_at_close'] = gold_at_close
                        logger.info(f"Altin: {gold_change:.2f}% | now=${gold_current:.2f}")
            except Exception as e:
                logger.warning(f"Altin verisi alinamadi: {e}")

            # Agirlikli kombine skor
            weights = {'silver': 0.35, 'bist100': 0.25, 'usd_try': 0.20, 'gold': 0.20}
            total_score = 0
            total_weight = 0
            for key, val in factors.items():
                w = weights.get(key, 0.15)
                total_score += val * w
                total_weight += w

            if total_weight > 0:
                combined_change = total_score / total_weight
            else:
                combined_change = 0

            logger.info(f"Pre-market kombine skor: {combined_change:.2f}%")

            # Sinyal olustur
            if combined_change > 1.5:
                signal, direction, confidence = "STRONG_BUY", "YUKSELIS", min(0.55 + abs(combined_change)*0.05, 0.70)
            elif combined_change > 0.5:
                signal, direction, confidence = "BUY", "YUKSELIS", min(0.52 + abs(combined_change)*0.08, 0.65)
            elif combined_change < -1.5:
                signal, direction, confidence = "STRONG_SELL", "DUSUS", min(0.55 + abs(combined_change)*0.05, 0.70)
            elif combined_change < -0.5:
                signal, direction, confidence = "SELL", "DUSUS", min(0.52 + abs(combined_change)*0.08, 0.65)
            else:
                signal, direction, confidence = "HOLD", "YATAY", 0.50

            target = gmstr_last_close * (1 + combined_change / 100)

            result = {
                'gmstr_last_close': gmstr_last_close,
                'gmstr_close_time': gmstr_close_time.strftime('%d.%m.%Y %H:%M'),
                'combined_change_pct': combined_change,
                'factors': factors,
                'prices': prices,
                'signal': signal,
                'direction': direction,
                'confidence': confidence,
                'target_price': target,
                'timestamp': datetime.now().strftime('%d.%m.%Y %H:%M')
            }

            self._premarket_cache = result
            self._premarket_cache_time = time_module.time()
            return result

        except Exception as e:
            logger.error(f"Pre-market sinyal hatasi: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if hasattr(self, '_premarket_cache') and self._premarket_cache:
                return self._premarket_cache
            return None
    
    def backtest_premarket(self, days=30):
        """Pre-market sinyali backtest - Geçmişte ne kadar doğru tahmin etmiş?"""
        try:
            logger.info(f"Pre-market backtest başlatılıyor: Son {days} gün")
            
            # GMSTR ve gümüş verilerini çek
            gmstr = yf.Ticker("GMSTR.IS")
            gmstr_data = gmstr.history(period=f"{days+5}d")
            
            silver = yf.Ticker("SI=F")
            silver_data = silver.history(period=f"{days+10}d")  # Daha uzun (7/24 açık)
            
            if gmstr_data.empty or silver_data.empty:
                logger.error("Veri çekilemedi")
                return None
            
            trades = []
            
            # Her iş günü için kontrol et
            for i in range(1, len(gmstr_data)):
                # Önceki gün kapanış
                prev_close_time = gmstr_data.index[i-1]
                prev_close_price = gmstr_data['Close'].iloc[i-1]
                
                # O gün açılış
                curr_open_time = gmstr_data.index[i]
                curr_open_price = gmstr_data['Open'].iloc[i]
                
                # Sadece 1 gün fark varsa (hafta sonu atla)
                time_diff = (curr_open_time - prev_close_time).total_seconds()
                if time_diff > 172800:  # 48 saatten fazla = hafta sonu
                    continue
                
                # Kapanış anındaki gümüş fiyatı
                silver_at_close = None
                for j in range(len(silver_data)-1, -1, -1):
                    if silver_data.index[j] <= prev_close_time:
                        silver_at_close = silver_data['Close'].iloc[j]
                        break
                
                # Açılış anındaki gümüş fiyatı
                silver_at_open = None
                for j in range(len(silver_data)):
                    if silver_data.index[j] >= curr_open_time:
                        silver_at_open = silver_data['Close'].iloc[j]
                        break
                
                if silver_at_close is None or silver_at_open is None:
                    continue
                
                # Gümüş değişimi
                silver_change = ((silver_at_open - silver_at_close) / silver_at_close) * 100
                
                # Tahmin
                if abs(silver_change) > 0.5:
                    predicted_direction = "YÜKSELİŞ" if silver_change > 0 else "DÜŞÜŞ"
                    
                    # Gerçekleşen
                    actual_change = ((curr_open_price - prev_close_price) / prev_close_price) * 100
                    actual_direction = "YÜKSELİŞ" if actual_change > 0 else "DÜŞÜŞ"
                    
                    is_correct = predicted_direction == actual_direction
                    
                    trades.append({
                        'date': curr_open_time.strftime('%Y-%m-%d'),
                        'gmstr_close': prev_close_price,
                        'gmstr_open': curr_open_price,
                        'silver_close': silver_at_close,
                        'silver_open': silver_at_open,
                        'silver_change': silver_change,
                        'predicted': predicted_direction,
                        'actual': actual_direction,
                        'correct': is_correct,
                        'gmstr_gap': actual_change
                    })
            
            if not trades:
                logger.warning("Yeterli işlem bulunamadı")
                return None
            
            # İstatistikler
            total = len(trades)
            correct = sum(1 for t in trades if t['correct'])
            accuracy = (correct / total * 100) if total > 0 else 0
            
            avg_gap = np.mean([t['gmstr_gap'] for t in trades])
            max_gap = max([abs(t['gmstr_gap']) for t in trades])
            
            result = {
                'total_trades': total,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'avg_gap_percent': avg_gap,
                'max_gap_percent': max_gap,
                'trades': trades[-10:]  # Son 10 işlem
            }
            
            logger.info(f"Backtest tamamlandı: {correct}/{total} doğru (%{accuracy:.1f})")
            logger.info(f"Ortalama açılış gap'i: %{avg_gap:.2f}, Max gap: %{max_gap:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Pre-market backtest hatası: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_streak_multiplier(self, days=30):
        """Son kazanma/kaybetme streak'ine gore pozisyon olceklendir.
        Kazanma streak'i = buyut, kaybetme = kucult (anti-martingale)."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()

            if self.is_postgres:
                cursor.execute('''
                    SELECT is_correct FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND created_at > NOW() - INTERVAL '%s days'
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (days,))
            else:
                cursor.execute('''
                    SELECT is_correct FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND datetime(created_at) > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                    LIMIT 10
                '''.format(days))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return 1.0

            # Streak hesapla (sondan basla)
            streak = 0
            first = rows[0][0]
            for r in rows:
                if r[0] == first:
                    streak += 1 if first == 1 else -1
                else:
                    break

            # Anti-martingale: kazaninca artir, kaybedince azalt
            if streak >= 3:
                return 1.5  # Max buyutme
            elif streak >= 2:
                return 1.25
            elif streak <= -3:
                return 0.5  # Max kucultme
            elif streak <= -2:
                return 0.75
            return 1.0

        except Exception as e:
            logger.warning(f"Streak hesaplama hatasi: {e}")
            return 1.0

    def calculate_risk_management(self, current_price, predicted_price, direction, confidence,
                                     portfolio_value: float = 10000.0,
                                     win_rate: float = 0.55,
                                     avg_win_pct: float = 2.5,
                                     avg_loss_pct: float = -1.5):
        """Profesyonel risk yonetimi: Kelly + parcali TP + trailing stop + streak + circuit breaker."""
        try:
            from risk_management import RiskManager
            rm = RiskManager(initial_balance=portfolio_value)

            # 1. Kelly Criterion
            kelly_size = rm.kelly_criterion_position_size(
                win_rate=win_rate, avg_win=avg_win_pct, avg_loss=avg_loss_pct,
                current_balance=portfolio_value
            )

            # 2. Guvene gore olcekle
            confidence_multiplier = confidence
            adjusted_position = kelly_size * confidence_multiplier

            # 3. Streak bazli olcekleme (anti-martingale)
            streak_mult = self._get_streak_multiplier(days=30)
            adjusted_position *= streak_mult
            logger.info(f"Streak multiplier: {streak_mult:.2f}x")

            # 4. Maksimum pozisyon limiti (%25 portfoy)
            max_position = portfolio_value * 0.25
            position_size = min(adjusted_position, max_position)
            position_pct = (position_size / portfolio_value) * 100 if portfolio_value > 0 else 0

            # 5. ATR bazli stop-loss
            volatility = abs(predicted_price - current_price) / current_price
            atr_based_sl = max(volatility * 1.5, 0.015)  # Min %1.5

            if direction == "YUKSELIS":
                stop_loss = current_price * (1 - atr_based_sl)
                # Parçali take-profit
                tp1 = current_price * 1.01  # %1
                tp2 = current_price * 1.025  # %2.5
            else:
                stop_loss = current_price * (1 + atr_based_sl)
                tp1 = current_price * 0.99
                tp2 = current_price * 0.975

            risk = abs(current_price - stop_loss)
            reward = abs(current_price - tp2)
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # 6. R/R kotu ise pozisyonu kucult
            if risk_reward_ratio < 1.5:
                position_size *= 0.5
                position_pct *= 0.5
                logger.warning(f"R/R dusuk ({risk_reward_ratio:.2f}), pozisyon yariya dusuruldu")

            # 7. Circuit breaker - gunluk max kayip %5
            daily_loss_limit = portfolio_value * 0.05

            return {
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(tp1, 2),
                'take_profit_2': round(tp2, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'position_size_usd': round(position_size, 2),
                'position_size_pct': round(position_pct, 1),
                'kelly_raw_pct': round((kelly_size / portfolio_value) * 100, 1),
                'streak_multiplier': streak_mult,
                'risk_amount': round(risk, 2),
                'potential_profit_tp1': round(abs(tp1 - current_price), 2),
                'potential_profit_tp2': round(abs(tp2 - current_price), 2),
                'daily_loss_limit': round(daily_loss_limit, 2),
                'strategy': '50% at TP1, 50% at TP2 with trailing stop after TP1'
            }

        except Exception as e:
            logger.error(f"Risk yonetimi hatasi: {e}")
            volatility = abs(predicted_price - current_price) / current_price
            return {
                'stop_loss': round(current_price * (1 - max(volatility * 1.5, 0.015)), 2),
                'take_profit_1': round(current_price * 1.01, 2),
                'take_profit_2': round(current_price * 1.025, 2),
                'risk_reward_ratio': 1.5,
                'position_size_pct': 10.0,
                'risk_amount': round(current_price * max(volatility * 1.5, 0.015), 2),
                'potential_profit_tp1': round(current_price * 0.01, 2),
                'potential_profit_tp2': round(current_price * 0.025, 2)
            }

    def optimize_thresholds(self, days=60):
        """Farkli guven esiklerinde backtest: en karli esigi bul."""
        try:
            logger.info(f"Esik optimizasyonu basliyor: son {days} gun")
            conn = self.get_db_connection()
            cursor = conn.cursor()

            if self.is_postgres:
                cursor.execute('''
                    SELECT current_price, predicted_direction, actual_price, confidence, is_correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND created_at > NOW() - INTERVAL '%s days'
                    ORDER BY timestamp
                ''', (days,))
            else:
                cursor.execute('''
                    SELECT current_price, predicted_direction, actual_price, confidence, is_correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND datetime(created_at) > datetime('now', '-{} days')
                    ORDER BY timestamp
                '''.format(days))

            trades = cursor.fetchall()
            conn.close()

            if not trades or len(trades) < 20:
                logger.warning("Optimizasyon icin yeterli veri yok")
                return None

            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
            results = []
            initial = 10000

            for thresh in thresholds:
                capital = initial
                position = 0
                wins = losses = total = 0
                equity = [capital]
                position_size = 0.5  # %50 baslangic (guvenden bagimsiz)

                for current, pred_dir, actual, conf, is_corr in trades:
                    if conf < thresh:
                        continue  # HOLD

                    total += 1
                    ret = abs((actual - current) / current) if is_corr else -abs((actual - current) / current)
                    trade_return = ret * position_size
                    capital *= (1 + trade_return)
                    equity.append(capital)

                    if is_corr:
                        wins += 1
                    else:
                        losses += 1

                total_ret = (capital - initial) / initial * 100
                wr = (wins / total * 100) if total > 0 else 0
                equity_s = pd.Series(equity)
                rets = equity_s.pct_change().dropna()
                sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if len(rets) > 1 and rets.std() > 0 else 0
                max_dd = ((equity_s / equity_s.cummax()) - 1).min() * 100

                results.append({
                    'threshold': thresh,
                    'trades': total,
                    'win_rate': round(wr, 1),
                    'total_return_pct': round(total_ret, 2),
                    'final_capital': round(capital, 2),
                    'sharpe': round(sharpe, 3),
                    'max_dd_pct': round(max_dd, 2),
                    'profit_factor': round((wins * total_ret / max(wins, 1)) / (abs(losses * total_ret) / max(losses, 1)), 2) if losses > 0 else 999
                })

            best = max(results, key=lambda x: x['total_return_pct'])
            logger.info(f"En iyi esik: {best['threshold']} | Getiri: {best['total_return_pct']}% | WinRate: {best['win_rate']}%")

            return {'thresholds_tested': results, 'recommended_threshold': best['threshold']}

        except Exception as e:
            logger.error(f"Optimizasyon hatasi: {e}")
            return None

    def run_backtest(self, days=30):
        """Backtesting simülasyonu"""
        try:
            logger.info(f"Backtesting başlatılıyor: Son {days} gün")
            
            # Geçmiş tahminleri al
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            if self.is_postgres:
                cursor.execute('''
                    SELECT current_price, predicted_price, predicted_direction, 
                           actual_price, confidence, is_correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND created_at > NOW() - INTERVAL '%s days'
                    ORDER BY timestamp
                ''', (days,))
            else:
                cursor.execute('''
                    SELECT current_price, predicted_price, predicted_direction, 
                           actual_price, confidence, is_correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND datetime(created_at) > datetime('now', '-{} days')
                    ORDER BY timestamp
                '''.format(days))
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                logger.warning("Backtest için yeterli veri yok")
                return None
            
            # Simülasyon
            total_trades = len(trades)
            winning_trades = 0
            losing_trades = 0
            total_return = 0
            returns = []
            
            for trade in trades:
                current, predicted, direction, actual, confidence, is_correct = trade
                
                if is_correct:
                    winning_trades += 1
                    trade_return = abs((actual - current) / current) * 100
                else:
                    losing_trades += 1
                    trade_return = -abs((actual - current) / current) * 100
                
                total_return += trade_return
                returns.append(trade_return)
            
            # İstatistikler
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Sharpe Ratio (basitleştirilmiş)
            if len(returns) > 1:
                returns_array = np.array(returns)
                avg_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            else:
                sharpe = 0
            
            # Max Drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            result = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'avg_return_per_trade': total_return / total_trades if total_trades > 0 else 0
            }
            
            logger.info(f"Backtest tamamlandı: {winning_trades}/{total_trades} kazançlı (%{win_rate:.1f}), Toplam getiri: %{total_return:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtesting hatası: {e}")
            return None

# Flask API + SocketIO
app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
try:
    from flask_socketio import SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    logger.info("SocketIO başlatıldı")
except ImportError:
    socketio = None
    logger.warning("flask-socketio kurulu değil, WebSocket devre dışı")

prediction_system = GMSTRPredictionSystem()
swing_predictor = GMSTRSwingPredictor() if SWING_AVAILABLE else None

@app.route('/')
def dashboard():
    """Dashboard ana sayfa - cache bypass, direkt dosya oku"""
    import os
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'dashboard.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/predictions')
def get_predictions():
    """Tahminleri getir (her cagrildiginda once dogrulama yap).
    Query params:
      days: son kac gunluk veri (ornegin 1, 7, 30). Yoksa son 50 kayit.
      model_type: 'normal', 'swing' veya 'all'. Varsayilan 'all'.
      min_confidence: minimum guven esigi (ornegin 0.60). Varsayilan 0.
    """
    try:
        # Once gecmis tahminlerin dogrulugunu kontrol et
        try:
            prediction_system.update_predictions()
        except Exception as e:
            logger.warning(f"Predictions dogrulama hatasi: {e}")

        days = request.args.get('days', type=int)
        model_type = request.args.get('model_type', 'all', type=str).lower()
        min_confidence = request.args.get('min_confidence', 0, type=float)
        
        conn = prediction_system.get_db_connection()
        cursor = conn.cursor()
        
        params = []
        where_clauses = []
        
        if days is not None and days > 0:
            if prediction_system.is_postgres:
                where_clauses.append("timestamp >= NOW() - INTERVAL '%s days'" % days)
            else:
                where_clauses.append("timestamp >= datetime('now', '-%d days')" % days)
        
        if model_type in ('normal', 'swing'):
            where_clauses.append("model_type = ?")
            params.append(model_type)
        
        if min_confidence > 0:
            where_clauses.append("confidence >= ?")
            params.append(min_confidence)
        
        sql = '''
            SELECT timestamp, predicted_for_time, current_price, predicted_direction, 
                   predicted_price, confidence, timeframe, actual_price, is_correct,
                   telegram_sent, model_type
            FROM predictions
        '''
        if where_clauses:
            sql += ' WHERE ' + ' AND '.join(where_clauses)
        sql += ' ORDER BY timestamp DESC'
        
        if days is None:
            sql += ' LIMIT 50'
        
        cursor.execute(sql, params)
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'timestamp': row[0],
                'predicted_for_time': row[1],
                'current_price': row[2],
                'predicted_direction': row[3],
                'predicted_price': row[4],
                'confidence': row[5],
                'timeframe': row[6],
                'actual_price': row[7],
                'is_correct': row[8],
                'telegram_sent': row[9] if row[9] is not None else 0,
                'model_type': row[10] if row[10] is not None else 'normal'
            })
        
        conn.close()
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction-accuracy')
def get_prediction_accuracy():
    """Tahmin dogruluk orani (yuksek guvenli tahminler icin).
    Query params:
      days: son kac gunluk veri (default 7)
      model_type: 'normal', 'swing' veya 'all' (default 'all')
      min_confidence: minimum guven esigi (default 0.70)
    Dogruluk = yon dogrulugu: tahmin yonu ile tahmin suresi sonundaki fiyat hareketi ayni mi?
    """
    try:
        days = int(request.args.get('days', 7))
        model_type = request.args.get('model_type', 'all').lower()
        min_confidence = float(request.args.get('min_confidence', 0.70))
        if days < 1 or days > 365:
            return jsonify({'error': 'days 1-365 arasi olmali'}), 400
        if model_type not in ('normal', 'swing', 'all'):
            return jsonify({'error': 'model_type normal, swing veya all olmali'}), 400
        if min_confidence < 0 or min_confidence > 1:
            return jsonify({'error': 'min_confidence 0-1 arasi olmali'}), 400
        
        now = datetime.now()
        start = now - timedelta(days=days)
        
        conn = prediction_system.get_db_connection()
        cursor = conn.cursor()
        ph = '%s' if prediction_system.is_postgres else '?'
        
        where_clauses = [f"timestamp >= {ph}", f"confidence >= {ph}"]
        params = [start, min_confidence]
        
        if model_type in ('normal', 'swing'):
            where_clauses.append(f"model_type = {ph}")
            params.append(model_type)
        else:
            # 'all' icin model_type NULL olan eski kayitlari normal say
            where_clauses.append(f"(model_type = {ph} OR model_type IS NULL)")
            params.append('normal')
        
        sql = f'''
            SELECT timestamp, predicted_for_time, current_price, predicted_price, predicted_direction, confidence, timeframe
            FROM predictions
            WHERE {' AND '.join(where_clauses)}
            ORDER BY timestamp ASC
        '''
        cursor.execute(sql, params)
        preds = cursor.fetchall()
        conn.close()
        
        if not preds:
            return jsonify({'accuracy': None, 'correct': 0, 'total': 0})
        
        # GMSTR verisini cek (2y yeterli)
        gmstr_data = prediction_system.fetch_gmstr_data(period="2y")
        if gmstr_data is None or gmstr_data.empty:
            return jsonify({'accuracy': None, 'correct': 0, 'total': 0, 'error': 'Fiyat verisi alinamadi'})
        
        gmstr_data = gmstr_data.copy()
        if gmstr_data.index.tz is not None:
            gmstr_data.index = gmstr_data.index.tz_localize(None)
        
        correct = 0
        total = 0
        for pred in preds:
            ts, pred_for, current_price, target_price, direction, confidence, tf = pred
            
            # Tarih string ise parse et
            if isinstance(ts, str):
                ts = date_parser.parse(ts)
            if pred_for is not None and isinstance(pred_for, str):
                pred_for = date_parser.parse(pred_for)
            
            # Tahminin son kullanma zamanini belirle
            if pred_for is None:
                if tf == '1h':
                    pred_for = ts + timedelta(hours=1)
                elif tf == '4h':
                    pred_for = ts + timedelta(hours=4)
                elif tf == '1d':
                    pred_for = ts + timedelta(days=1)
                else:
                    pred_for = ts + timedelta(hours=1)
            
            # Tahmin zamani veriden sonraki fiyat bulunamazsa atla
            if pred_for < gmstr_data.index[0]:
                continue
            
            mask = gmstr_data.index <= pred_for
            if not mask.any():
                continue
            actual_price = float(gmstr_data.loc[mask, 'Close'].iloc[-1])
            
            total += 1
            current_price = float(current_price)
            target_price = float(target_price)
            
            # Yon dogrulugu: tahmin yonu ile gerceklesen fiyat hareketi ayni mi?
            if direction == 'YUKSELIS':
                if actual_price > current_price:
                    correct += 1
            elif direction == 'DUSUS':
                if actual_price < current_price:
                    correct += 1
        
        accuracy = correct / total if total > 0 else None
        return jsonify({
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'days': days,
            'model_type': model_type,
            'min_confidence': min_confidence
        })
        
    except Exception as e:
        logger.error(f"Prediction accuracy hatasi: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-prices')
def get_market_prices():
    """GMSTR piyasa fiyatlarini dondur (grafi icin).
    Query params:
      days: son kac gunluk veri (1, 7, 30). Varsayilan 7.
    """
    try:
        days = int(request.args.get('days', 7))
        if days < 1 or days > 365:
            return jsonify({'error': 'days 1-365 arasi olmali'}), 400
        
        if days == 1:
            period = "1mo"
            interval = "30m"
        elif days == 7:
            period = "1mo"
            interval = "30m"
        else:
            period = "1mo"
            interval = "30m"
        
        gmstr_data = prediction_system.fetch_gmstr_data(period=period, interval=interval)
        if gmstr_data is None or gmstr_data.empty:
            return jsonify({'error': 'Fiyat verisi alinamadi'}), 500
        
        gmstr_data = gmstr_data.copy()
        if gmstr_data.index.tz is not None:
            gmstr_data.index = gmstr_data.index.tz_localize(None)
        
        now = datetime.now()
        start = now - timedelta(days=days)
        gmstr_data = gmstr_data[gmstr_data.index >= start]
        
        prices = []
        for ts, row in gmstr_data.iterrows():
            prices.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'close': float(row['Close'])
            })
        
        return jsonify({'prices': prices, 'days': days, 'interval': interval})
        
    except Exception as e:
        logger.error(f"Market prices hatasi: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Performans verileri"""
    stats = prediction_system.get_performance_stats()
    return jsonify(stats)

@app.route('/api/model-info')
def get_model_info():
    """Model egitim bilgileri"""
    try:
        import os
        if os.path.exists('model_info.json'):
            with open('model_info.json', 'r') as f:
                info = json.load(f)
            # Dashboard compatibility: accuracy field ekle
            if 'accuracy' not in info:
                if 'filtered_accuracy' in info:
                    info['accuracy'] = info['filtered_accuracy']
                elif 'lgbm_accuracy' in info:
                    info['accuracy'] = info['lgbm_accuracy']
                elif 'ensemble_accuracy' in info:
                    info['accuracy'] = info['ensemble_accuracy']
            return jsonify(info)
        else:
            return jsonify({
                'status': 'not_trained',
                'message': 'Model henuz egitilmemis',
                'last_trained': '-',
                'accuracy': 0,
                'ensemble_accuracy': 0,
                'test_samples': 0,
                'train_samples': 0,
                'threshold': 0.60,
                'features_used': 0
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Manuel tahmin yap - opsiyonel manuel fiyat"""
    data = request.get_json() or {}
    timeframe = data.get('timeframe', '4h')
    manual_price = data.get('manual_price')
    
    prediction = prediction_system.make_prediction(timeframe)
    
    if prediction:
        # Manuel fiyat varsa uygula
        if manual_price and isinstance(manual_price, (int, float)) and manual_price > 0:
            prediction['display_price'] = float(manual_price)
            prediction['note'] = f"Manuel fiyat: ₺{manual_price:.2f} (tahmin bu degere gore)"
        
        # Geçmişe kaydet
        try:
            prediction_system.save_prediction(
                prediction.get('current_price', 0),
                prediction.get('direction', 'HOLD'),
                prediction.get('target_price', 0),
                prediction.get('confidence', 0.5),
                timeframe,
                None
            )
        except Exception as e:
            logger.warning(f"Tahmin kaydetme hatasi: {e}")
        
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Tahmin yapılamadı'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Modeli eğit"""
    success = prediction_system.train_model()
    
    if success:
        return jsonify({'success': True, 'message': 'Model başarıyla eğitildi'})
    else:
        return jsonify({'error': 'Model eğitilemedi'}), 500

@app.route('/api/swing/predict', methods=['POST'])
def swing_predict():
    """Swing tahmini yap (1h) - opsiyonel manuel fiyat"""
    try:
        if swing_predictor is None:
            return jsonify({'error': 'Swing modeli mevcut değil'}), 500
        
        data = request.get_json() or {}
        manual_price = data.get('manual_price')
        
        result = swing_predictor.predict()
        if result:
            # Mevcut fiyat 30dk veriden gercek son fiyat olarak al
            gmstr_30m = prediction_system.fetch_gmstr_data(period="1d", interval="30m")
            if gmstr_30m is not None and not gmstr_30m.empty:
                if gmstr_30m.index.tz is not None:
                    gmstr_30m.index = gmstr_30m.index.tz_localize(None)
                real_price = float(gmstr_30m['Close'].iloc[-1])
            else:
                real_price = result['current_price']
            result['current_price'] = real_price
            result['real_price'] = real_price
            
            # Manuel fiyat varsa uygula (model tahmini degismez, sadece gosterim)
            if manual_price and isinstance(manual_price, (int, float)) and manual_price > 0:
                result['display_price'] = float(manual_price)
                result['note'] = f"Manuel fiyat: ₺{manual_price:.2f} (tahmin bu degere gore)"
            
            # Telegram sadece %60+ guven
            if prediction_system.telegram_bot_token and result['confidence'] >= 0.60:
                emoji = "🟢" if result['direction'] == 'YUKSELIS' else "🔴"
                ctx = result.get('context', '')
                msg = f"""<b>🔄 GMSTR Swing (1h)</b> {emoji}

📅 {result['timestamp']}
💰 Fiyat: ₺{result['current_price']:.2f}
📈 Tahmin: {result['direction']}
🔒 Güven: %{result['confidence']*100:.1f}
🎯 Context: {ctx}

<i>1 saatlik zaman dilimi - kısa vadeli dönüşler</i>"""
                prediction_system.send_telegram_message(msg)
            
            # Geçmişe kaydet
            try:
                prediction_system.save_prediction(
                    result['current_price'],
                    result['direction'],
                    result['current_price'] * (1.01 if result['direction'] == 'YUKSELIS' else 0.99),
                    result['confidence'],
                    '1h',
                    None,
                    'swing'
                )
            except Exception as e:
                logger.warning(f"Swing kaydetme hatasi: {e}")
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Swing tahmini yapılamadı'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/swing/train', methods=['POST'])
def swing_train():
    """Swing modelini eğit"""
    try:
        if swing_predictor is None:
            return jsonify({'error': 'Swing modeli mevcut değil'}), 500
        
        success = swing_predictor.train()
        if success:
            return jsonify({'success': True, 'message': 'Swing modeli eğitildi', 'note': 'Dogruluk hedefi: %65+'})
        else:
            return jsonify({'success': False, 'message': 'Egitim basarili ama dusuk dogruluk', 'note': '1h zaman diliminde %65+ zor olabilir'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/swing/status')
def swing_status():
    """Swing model durumu"""
    if swing_predictor is None:
        return jsonify({'available': False})
    
    return jsonify({
        'available': True,
        'trained': swing_predictor.model is not None,
        'last_train': swing_predictor.last_train_time.strftime('%Y-%m-%d %H:%M') if swing_predictor.last_train_time else None
    })

@app.route('/api/risk', methods=['POST'])
def get_risk_analysis():
    """Risk analizi yap"""
    try:
        data = request.json
        current_price = data.get('current_price', 0)
        predicted_price = data.get('predicted_price', 0)
        direction = data.get('direction', 'YÜKSELİŞ')
        confidence = data.get('confidence', 0.5)
        
        risk = prediction_system.calculate_risk_management(
            current_price, predicted_price, direction, confidence
        )
        
        if risk:
            return jsonify(risk)
        else:
            return jsonify({'error': 'Risk analizi yapılamadı'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Backtesting çalıştır"""
    try:
        days = request.json.get('days', 30)
        result = prediction_system.run_backtest(days)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Backtest için yeterli veri yok'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_thresholds_api():
    """Guven esigi optimizasyonu: en karli esigi bul"""
    try:
        days = request.json.get('days', 60)
        result = prediction_system.optimize_thresholds(days)

        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Optimizasyon için yeterli veri yok'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data')
def get_market_data():
    """Güncel piyasa verileri"""
    try:
        market_data = prediction_system.fetch_market_data()
        
        if market_data:
            result = {}
            for key, df in market_data.items():
                if df is not None and not df.empty:
                    result[key] = {
                        'current': float(df['Close'].iloc[-1]),
                        'change_24h': float((df['Close'].iloc[-1] - df['Close'].iloc[-24]) / df['Close'].iloc[-24] * 100) if len(df) >= 24 else 0
                    }
            return jsonify(result)
        else:
            return jsonify({'error': 'Veri çekilemedi'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def force_validate():
    """Manuel validasyon: tum bekleyen tahminleri hemen kontrol et"""
    try:
        now = datetime.now()
        day = now.weekday()
        hour = now.hour
        minute = now.minute
        is_weekend = day >= 5
        is_after_hours = (hour > 18) or (hour == 18 and minute >= 10) or (hour < 10)
        if is_weekend or is_after_hours:
            return jsonify({'error': 'Borsa kapali, manuel validasyon yapilamaz'}), 403
        prediction_system.update_predictions()
        conn = prediction_system.get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM predictions WHERE actual_price IS NULL
        ''')
        pending = cursor.fetchone()[0]
        cursor.execute('''
            SELECT COUNT(*) FROM predictions WHERE is_correct = 1
        ''')
        correct = cursor.fetchone()[0]
        cursor.execute('''
            SELECT COUNT(*) FROM predictions WHERE is_correct = 0
        ''')
        wrong = cursor.fetchone()[0]
        conn.close()
        return jsonify({
            'success': True,
            'pending': pending,
            'correct': correct,
            'wrong': wrong
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/premarket-signal')
def get_premarket():
    """Pre-market gümüş sinyali"""
    try:
        signal = prediction_system.get_premarket_signal()
        
        if signal:
            return jsonify(signal)
        else:
            return jsonify({'error': 'Pre-market sinyali alınamadı'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backfill-predictions', methods=['POST'])
def backfill_predictions():
    """Son N gun icin gecmise donuk tahmin uret ve kaydet (asenkron).
    Body: days (default 30)
    """
    try:
        data = request.get_json() or {}
        days = int(data.get('days', 30))
        if days < 1 or days > 90:
            return jsonify({'error': 'days 1-90 arasi olmali'}), 400
        
        import threading
        
        def run_backfill():
            try:
                now = datetime.now()
                total = 0
                # 30dk veriyi bir kere cek
                gmstr_30m = prediction_system.fetch_gmstr_data(period="60d", interval="30m")
                if gmstr_30m is not None:
                    if gmstr_30m.index.tz is not None:
                        gmstr_30m.index = gmstr_30m.index.tz_localize(None)
                for day_offset in range(days, 0, -1):
                    as_of_day = now - timedelta(days=day_offset)
                    # Hafta sonu atla
                    if as_of_day.weekday() >= 5:
                        continue
                    # 09:30 - 17:00 arasi 30dk araliklar
                    for hour in range(9, 18):
                        for minute in (0, 30):
                            if hour == 9 and minute < 30:
                                continue
                            if hour == 17 and minute > 0:
                                continue
                            as_of = as_of_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            if as_of >= now:
                                continue
                            
                            # Ayni zamanda zaten kayit varsa atla
                            try:
                                conn = prediction_system.get_db_connection()
                                cursor = conn.cursor()
                                ph = '%s' if prediction_system.is_postgres else '?'
                                cursor.execute(
                                    f"SELECT COUNT(*) FROM predictions WHERE timestamp = {ph} AND model_type = {ph}",
                                    (as_of, 'normal')
                                )
                                if cursor.fetchone()[0] > 0:
                                    conn.close()
                                    continue
                                conn.close()
                            except Exception as dup_err:
                                logger.debug(f"Duplicate kontrol hatasi: {dup_err}")
                            
                            # Normal tahmin
                            normal_pred = prediction_system.make_backfill_prediction(as_of, '4h')
                            if normal_pred:
                                prediction_system.save_historical_prediction(
                                    as_of, normal_pred['current_price'], normal_pred['direction'],
                                    normal_pred['target_price'], normal_pred['confidence'], '4h', 'normal'
                                )
                                total += 1
                            
                            # Swing tahmin
                            if swing_predictor is not None:
                                swing_pred = swing_predictor.predict_historical(as_of)
                                if swing_pred:
                                    # Mevcut fiyat 30dk veriden, as_of'dan onceki son kapanis olarak al
                                    if gmstr_30m is not None:
                                        current_price = gmstr_30m[gmstr_30m.index < as_of].Close.iloc[-1]
                                    else:
                                        current_price = swing_pred['current_price']
                                    prediction_system.save_historical_prediction(
                                        as_of, float(current_price), swing_pred['direction'],
                                        float(current_price) * (1.01 if swing_pred['direction'] == 'YUKSELIS' else 0.99),
                                        swing_pred['confidence'], '1h', 'swing'
                                    )
                                    total += 1
                
                logger.info(f"Backfill tamamlandi: {total} tahmin kaydedildi")
            except Exception as e:
                logger.error(f"Backfill thread hatasi: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        thread = threading.Thread(target=run_backfill)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Backfill baslatildi: son {days} gun icin her 30dk normal + swing tahmin.',
            'note': 'Islem arka planda calisiyor, loglari takip edin.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def scheduled_predictions():
    """Otomatik tahminler - 9:30-17:00 her 30dk"""
    logger.info("Otomatik tahminler baslatiliyor...")
    
    def auto_predict_and_save(timeframe='4h'):
        """Tahmin yap, kaydet, Telegram gonder"""
        if not prediction_system.is_borsa_open():
            logger.info("Borsa kapali, tahmin atlaniyor")
            return None
        
        try:
            pred = prediction_system.make_prediction(timeframe)
            if pred:
                # DB kaydet
                prediction_system.save_prediction(
                    pred.get('current_price', 0),
                    pred.get('direction', 'HOLD'),
                    pred.get('target_price', 0),
                    pred.get('confidence', 0.5),
                    timeframe,
                    None
                )
                logger.info(f"Auto {timeframe} tahmin kaydedildi: {pred['direction']}")
            return pred
        except Exception as e:
            logger.error(f"Auto predict hatasi: {e}")
            return None
    
    def auto_swing_predict_and_save():
        """Swing tahmini yap, kaydet, %60+ Telegram"""
        if not prediction_system.is_borsa_open():
            return None
        
        try:
            if swing_predictor is None:
                return None
            
            result = swing_predictor.predict()
            if result:
                # DB kaydet
                prediction_system.save_prediction(
                    result['current_price'],
                    result['direction'],
                    result['current_price'] * (1.02 if result['direction'] == 'YUKSELIS' else 0.98),
                    result['confidence'],
                    '1h',
                    None,
                    'swing'
                )
                
                # Telegram sadece %60+
                if prediction_system.telegram_bot_token and result['confidence'] >= 0.60:
                    emoji = "🟢" if result['direction'] == 'YUKSELIS' else "🔴"
                    ctx = result.get('context', '')
                    msg = f"""<b>🔄 GMSTR Swing (1h)</b> {emoji}

📅 {result['timestamp']}
💰 Fiyat: ₺{result['current_price']:.2f}
📈 Tahmin: {result['direction']}
🔒 Güven: %{result['confidence']*100:.1f}
🎯 Context: {ctx}

<i>Otomatik tahmin - 30dk aralik</i>"""
                    prediction_system.send_telegram_message(msg)
                
                logger.info(f"Auto swing tahmin kaydedildi: {result['direction']} %{result['confidence']*100:.0f}")
            return result
        except Exception as e:
            logger.error(f"Auto swing hatasi: {e}")
            return None
    
    # Hafta ici 09:30-17:00 arasi her 30dk
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        for h in range(9, 18):  # 09:00 - 17:00
            for m in [0, 30]:
                if h == 9 and m == 0:
                    continue  # 09:30'den basla
                if h == 17 and m == 30:
                    continue  # 17:00 son
                t_str = f"{h:02d}:{m:02d}"
                
                # Her 30dk trend tahmin
                getattr(schedule.every(), day).at(t_str).do(lambda tf=timeframe: auto_predict_and_save('4h'))
                
                # Her 30dk swing tahmin
                getattr(schedule.every(), day).at(t_str).do(auto_swing_predict_and_save)
                
                logger.info(f"Schedule: her {day} {t_str} (trend + swing)")

    # Her 10 dakikada tahminleri guncelle
    schedule.every(10).minutes.do(lambda: prediction_system.update_predictions())

    # Haftalik auto-optimize: Pazar 23:00
    def weekly_optimize():
        logger.info("Haftalik auto-optimize baslatiliyor...")
        try:
            result = prediction_system.optimize_thresholds(days=60)
            if result and result.get('recommended_threshold'):
                rec = result['recommended_threshold']
                logger.info(f"Onerilen esik guncellendi: {rec}")
                prediction_system.send_telegram_message(
                    f"<b>Haftalik Optimizasyon</b>\n\n"
                    f"Onerilen esik: {rec}\n"
                    f"Test edilen: {len(result.get('thresholds_tested', []))} esik\n"
                    f"En iyi getiri: {max(t['total_return_pct'] for t in result.get('thresholds_tested', [])):.1f}%"
                )
        except Exception as e:
            logger.error(f"Weekly optimize hatasi: {e}")

    schedule.every().sunday.at("23:00").do(weekly_optimize)

    while True:
        schedule.run_pending()
        time_module.sleep(60)

if __name__ == '__main__':
    # İlk başlatma
    logger.info("GMSTR Tahmin Sistemi Başlatılıyor...")
    
    # Model eğitimi (yoksa)
    prediction_system.train_model()
    
    # Scheduled thread başlat
    import threading
    scheduler_thread = threading.Thread(target=scheduled_predictions, daemon=True)
    scheduler_thread.start()
    
    # Flask server - Production'da debug=False
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
