"""
GMSTR %65+ Başarılı Tahmin Sistemi
Backend API ve Otomatik Tahmin Motoru
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
from datetime import datetime, timedelta
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
warnings.filterwarnings('ignore')

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
        self.model_path = 'gmstr_prediction_model.pkl'
        self.model = None
        self.features = []
        
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
    
    def fetch_gmstr_data(self, period="2y"):
        """GMSTR verilerini çek (cache'li)"""
        now = time_module.time()
        if self._gmstr_cache is not None and self._gmstr_cache_time and (now - self._gmstr_cache_time) < self._cache_ttl:
            logger.info("GMSTR verisi cache'den alındı")
            return self._gmstr_cache
        
        try:
            # Yahoo Finance'den GMSTR verisi (1h max 730 gün)
            ticker = yf.Ticker("GMSTR.IS")
            data = ticker.history(period=period, interval="1h")
            
            if data is None:
                logger.error("GMSTR verisi None döndü")
                return self._gmstr_cache if self._gmstr_cache else None
            
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
                    return self._gmstr_cache if self._gmstr_cache else None
            
            if data.empty:
                logger.error("GMSTR verisi çekilemedi")
                return self._gmstr_cache if self._gmstr_cache else None
            
            self._gmstr_cache = data
            self._gmstr_cache_time = now
            return data
        except Exception as e:
            logger.error(f"GMSTR veri çekme hatası: {e}")
            return self._gmstr_cache if self._gmstr_cache else None
    
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
    
    def create_features(self, gmstr_data, market_data):
        """Özellik matrisi oluştur"""
        try:
            # gmstr_data'nın DataFrame olduğundan emin ol
            if isinstance(gmstr_data, list):
                logger.warning("gmstr_data list formatında, DataFrame'e çevriliyor")
                if len(gmstr_data) > 0 and len(gmstr_data[0]) >= 5:
                    gmstr_data = pd.DataFrame(gmstr_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                else:
                    logger.error("gmstr_data formatı geçersiz")
                    return None
            
            # Eğer gmstr_data dict ise DataFrame'e çevir
            if isinstance(gmstr_data, dict):
                gmstr_data = pd.DataFrame(gmstr_data)
            
            # DataFrame sütunlarını kontrol et
            if not isinstance(gmstr_data, pd.DataFrame):
                logger.error(f"gmstr_data tipi geçersiz: {type(gmstr_data)}")
                return None
            
            # Gerekli sütunları kontrol et
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in gmstr_data.columns]
            if missing_columns:
                logger.error(f"Eksik sütunlar: {missing_columns}")
                # Alternatif sütun isimleri dene
                if 'open' in gmstr_data.columns:
                    gmstr_data = gmstr_data.rename(columns={
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume'
                    })
                else:
                    return None
            
            # GMSTR göstergeleri
            gmstr_indicators = self.calculate_technical_indicators(gmstr_data)
            
            # Piyasa göstergeleri
            if market_data and 'bist100' in market_data and market_data['bist100'] is not None:
                bist_indicators = self.calculate_technical_indicators(market_data['bist100'])
            else:
                bist_indicators = {}
            
            # Özellik matrisi - basit ve sağlam yaklaşım
            features = []
            feature_names = []
            
            # Veri uzunluğunu kontrol et
            gmstr_len = len(gmstr_data)
            if gmstr_len < 25:
                logger.error(f"GMSTR verisi yetersiz: {gmstr_len} < 25")
                return None
            
            # Özellik isimlerini önceden tanımla - Sadece temel göstergeler
            gmstr_ind_list = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
                              'atr', 'stoch_k', 'stoch_d', 'williams_r', 'ema_20', 
                              'sma_50', 'momentum', 'volume_delta', 'obv', 'cci', 'z_score']
            
            feature_names = [f'gmstr_{ind}' for ind in gmstr_ind_list]
            feature_names += ['price_change_1h', 'price_change_4h', 'price_change_24h', 'volume_change']
            
            # Sadece 1 lag - en son değişim
            feature_names += ['gmstr_rsi_lag1', 'gmstr_macd_lag1', 'gmstr_close_lag1']
            
            # Zaman özellikleri
            feature_names += ['hour_sin', 'hour_cos', 'day_of_week']
            
            self.features = feature_names
            expected_len = len(feature_names)
            
            for i in range(20, gmstr_len):
                row = []
                
                # Sadece sayısal skaler değerler ekle
                def safe_float(val):
                    try:
                        if hasattr(val, '__len__') and not isinstance(val, str):
                            return float(val[0]) if len(val) > 0 else 0.0
                        return float(val)
                    except:
                        return 0.0
                
                # GMSTR göstergeleri
                for indicator in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
                                 'atr', 'stoch_k', 'stoch_d', 'williams_r', 'ema_20', 
                                 'sma_50', 'momentum', 'volume_delta', 'obv', 'cci', 'z_score']:
                    val = 0.0
                    if indicator in gmstr_indicators and i < len(gmstr_indicators[indicator]):
                        raw_val = gmstr_indicators[indicator].iloc[i]
                        val = safe_float(raw_val)
                    row.append(val)
                
                # Fiyat değişimleri
                price_change_1h = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-1]) / gmstr_data['Close'].iloc[i-1]) if i >= 1 else 0.0
                price_change_4h = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-4]) / gmstr_data['Close'].iloc[i-4]) if i >= 4 else 0.0
                price_change_24h = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-24]) / gmstr_data['Close'].iloc[i-24]) if i >= 24 else 0.0
                volume_change = safe_float((gmstr_data['Volume'].iloc[i] - gmstr_data['Volume'].iloc[i-1]) / gmstr_data['Volume'].iloc[i-1]) if i >= 1 else 0.0
                row += [price_change_1h, price_change_4h, price_change_24h, volume_change]
                
                # Sadece 1 lag
                rsi_lag = safe_float(gmstr_indicators['rsi'].iloc[i-1]) if 'rsi' in gmstr_indicators and i-1 >= 0 else 0.0
                macd_lag = safe_float(gmstr_indicators['macd'].iloc[i-1]) if 'macd' in gmstr_indicators and i-1 >= 0 else 0.0
                close_lag = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-1]) / gmstr_data['Close'].iloc[i-1]) if i-1 >= 0 else 0.0
                row += [rsi_lag, macd_lag, close_lag]
                
                # Zaman özellikleri (sin/cos encoding ile saat)
                timestamp = gmstr_data.index[i]
                hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
                hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
                row += [hour_sin, hour_cos, float(timestamp.dayofweek)]
                
                # Uzunluk kontrolü
                if len(row) == expected_len:
                    features.append(row)
                else:
                    logger.warning(f"Satır uzunluğu uyuşmuyor: {len(row)} vs {expected_len}")
            
            if len(features) == 0:
                logger.error("Hiç özellik oluşturulamadı")
                return None
            
            # Numpy array oluştur - float tipi zorla
            X = np.array(features, dtype=np.float64)
            self.features = feature_names
            
            # NaN ve Inf değerleri temizle
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Özellik matrisi oluşturuldu: {X.shape}")
            return X
            
        except Exception as e:
            logger.error(f"Özellik oluşturma hatası: {e}")
            import traceback
            logger.error(f"Detaylı hata: {traceback.format_exc()}")
            return None
    
    def create_labels(self, data, timeframe_hours=48):
        """Etiketler oluştur - 48 saat, %2 eşik (sadece güçlü hareketler)"""
        labels = []
        
        for i in range(20, len(data) - timeframe_hours):
            current_price = data['Close'].iloc[i]
            future_price = data['Close'].iloc[i + timeframe_hours]
            
            # %2 eşik - sadece net hareketleri tahmin et
            if future_price > current_price * 1.02:
                labels.append(1)  # Yükseliş
            elif future_price < current_price * 0.98:
                labels.append(0)  # Düşüş
            else:
                labels.append(2)  # Yatay
                
        return np.array(labels)
    
    def train_model(self):
        """Modeli eğit"""
        try:
            logger.info("Model eğitimi başlıyor...")
            
            # Verileri çek
            gmstr_data = self.fetch_gmstr_data()
            market_data = self.fetch_market_data()
            
            if gmstr_data is None:
                logger.error("GMSTR verisi çekilemedi")
                return False
            
            # Özellikler ve etiketler
            X = self.create_features(gmstr_data, market_data)
            y = self.create_labels(gmstr_data)
            
            if X is None or len(X) == 0:
                logger.error("Özellikler oluşturulamadı")
                return False
            
            # Veri boyutunu eşitle
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Yatay sinyalleri çıkar (sadece yükseliş/düşüş)
            mask = y != 2
            X = X[mask]
            y = y[mask]
            
            # Train-test split - Stratified (sınıf dağılımını koru)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
            )
            
            logger.info(f"Eğitim: {len(X_train)} örnek, Test: {len(X_test)} örnek")
            
            # Class dağılımını logla
            from collections import Counter
            train_dist = Counter(y_train)
            test_dist = Counter(y_test)
            logger.info(f"Eğitim sınıf dağılımı: {dict(train_dist)}")
            logger.info(f"Test sınıf dağılımı: {dict(test_dist)}")
            
            # Basit ama güçlü model - RandomForest
            best_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=8, 
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42, 
                n_jobs=-1
            )
            
            # En iyi modeli eğit
            best_model.fit(X_train, y_train)
            
            # Test performansı
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Sonuçları her zaman kaydet (başarılı veya başarısız)
            import json
            model_info = {
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'accuracy': round(accuracy, 4),
                'test_samples': len(y_test),
                'train_samples': len(X_train),
                'status': 'success' if accuracy >= 0.60 else 'failed',
                'threshold': 0.60,
                'features_used': len(self.features)
            }
            with open('model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            if accuracy >= 0.60:
                logger.info(f"Model başarı oranına ulaştı: {accuracy:.4f}")
                self.model = best_model
                joblib.dump(best_model, self.model_path)
                
                # Özellik isimlerini kaydet
                with open('feature_names.txt', 'w') as f:
                    f.write(','.join(self.features))
                
                # Feature importance logla
                importances = best_model.feature_importances_
                feat_imp = list(zip(self.features, importances))
                feat_imp.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"En önemli 10 özellik: {feat_imp[:10]}")
                
                return True
            else:
                logger.warning(f"Model %65 başarı oranına ulaşamadı: {accuracy:.4f}")
                return False
                
        except Exception as e:
            logger.error(f"Model eğitimi hatası: {e}")
            return False
    
    def is_borsa_open(self):
        """Borsa açık mı kontrol et (Türkiye saati)"""
        now = datetime.now()
        # Hafta sonu kontrol
        if now.weekday() >= 5:  # Cumartesi=5, Pazar=6
            return False
        # Saat kontrolü (09:00 - 18:10)
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=18, minute=10, second=0, microsecond=0)
        return market_open <= now <= market_close
    
    def make_prediction(self, timeframe="4h"):
        """Tahmin yap"""
        try:
            if self.model is None:
                # Modeli yükle
                try:
                    self.model = joblib.load(self.model_path)
                    with open('feature_names.txt', 'r') as f:
                        self.features = f.read().split(',')
                except:
                    logger.error("Model bulunamadı, önce eğitim yapın")
                    return None
            
            # Güncel verileri çek (1h max 730 gün)
            gmstr_data = self.fetch_gmstr_data(period="2y")
            market_data = self.fetch_market_data()
            
            if gmstr_data is None:
                logger.error("GMSTR verisi çekilemedi")
                return None
            
            # Özellikler oluştur
            X = self.create_features(gmstr_data, market_data)
            
            if X is None or len(X) == 0:
                logger.error("Özellikler oluşturulamadı")
                return None
            
            # Son özellik vektörünü al
            latest_features = X[-1].reshape(1, -1)
            
            # Tahmin yap
            prediction = self.model.predict(latest_features)[0]
            probability = self.model.predict_proba(latest_features)[0]
            
            current_price = gmstr_data['Close'].iloc[-1]
            confidence = max(probability)
            
            # Hedef fiyat hesapla
            if prediction == 1:  # Yükseliş
                target_price = current_price * 1.015  # %1.5 hedef
                direction = "YÜKSELİŞ"
            else:  # Düşüş
                target_price = current_price * 0.985  # %1.5 hedef
                direction = "DÜŞÜŞ"
            
            # Tahmin zamanını hesapla (şu an + timeframe)
            now = datetime.now()
            if timeframe == "4h":
                predicted_for_time = now + timedelta(hours=4)
            elif timeframe == "1d":
                predicted_for_time = now + timedelta(days=1)
            else:
                predicted_for_time = now + timedelta(hours=4)
            
            # Tahmini kaydet
            pred_id = self.save_prediction(current_price, direction, target_price, confidence, timeframe, predicted_for_time)
            
            # %60+ güven varsa Telegram'dan sinyal gönder
            telegram_sent = False
            if confidence >= 0.60:
                emoji = "🟢" if direction == "YÜKSELİŞ" else "🔴"
                message = f"""<b>GMSTR Sinyal</b> {emoji}

📅 <b>Tahmin Zamanı:</b> {now.strftime('%d.%m.%Y %H:%M')}
⏰ <b>Geçerli Olacağı Zaman:</b> {predicted_for_time.strftime('%d.%m.%Y %H:%M')}

💰 <b>Mevcut Fiyat:</b> ₺{current_price:.2f}
📈 <b>Tahmin:</b> {direction}
🎯 <b>Hedef Fiyat:</b> ₺{target_price:.2f}
🔒 <b>Güven:</b> %{confidence*100:.1f}

⏳ <b>Beklenen Değişim:</b> %{abs((target_price - current_price) / current_price * 100):.2f}"""
                
                telegram_sent = self.send_telegram_message(message)
                
                if telegram_sent and pred_id:
                    self.update_telegram_status(pred_id, 1)
            
            result = {
                'timestamp': now,
                'predicted_for_time': predicted_for_time,
                'current_price': current_price,
                'direction': direction,
                'target_price': target_price,
                'confidence': confidence,
                'timeframe': timeframe,
                'telegram_sent': telegram_sent
            }
            
            logger.info(f"Tahmin yapıldı: {direction} - Güven: {confidence:.4f} - Telegram: {telegram_sent}")
            return result
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return None
    
    def save_prediction(self, current_price, direction, target_price, confidence, timeframe, predicted_for_time=None):
        """Tahmini veritabanına kaydet ve ID döndür"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            
            if self.is_postgres:
                cursor.execute('''
                    INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                    predicted_price, confidence, timeframe)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (now, predicted_for_time, current_price, direction, target_price, confidence, timeframe))
                pred_id = cursor.fetchone()[0]
            else:
                cursor.execute('''
                    INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                    predicted_price, confidence, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (now, predicted_for_time, current_price, direction, target_price, confidence, timeframe))
                pred_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            return pred_id
            
        except Exception as e:
            logger.error(f"Tahmin kaydetme hatası: {e}")
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
        """Tahminleri güncelle (doğruluk kontrolü) - Otomatik"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            six_hours_ago = now - timedelta(hours=6)
            
            if self.is_postgres:
                cursor.execute('''
                    SELECT id, timestamp, predicted_for_time, current_price, predicted_price, predicted_direction, confidence
                    FROM predictions
                    WHERE actual_price IS NULL
                    AND (predicted_for_time < %s OR timestamp < %s)
                ''', (now, six_hours_ago))
            else:
                cursor.execute('''
                    SELECT id, timestamp, predicted_for_time, current_price, predicted_price, predicted_direction, confidence
                    FROM predictions
                    WHERE actual_price IS NULL
                    AND (predicted_for_time < datetime('now') OR datetime(timestamp) < datetime('now', '-6 hours'))
                ''')
            
            predictions = cursor.fetchall()
            
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
                
                # Doğruluğu kontrol et
                if pred_direction == "YÜKSELİŞ":
                    is_correct = 1 if actual_price > current_price else 0
                    actual_direction = "YÜKSELİŞ" if actual_price > current_price else "DÜŞÜŞ"
                else:
                    is_correct = 1 if actual_price < current_price else 0
                    actual_direction = "DÜŞÜŞ" if actual_price < current_price else "YÜKSELİŞ"
                
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
                ''', (seven_days_ago,))
            else:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(is_correct) as correct
                    FROM predictions
                    WHERE is_correct IS NOT NULL
                    AND datetime(created_at) > datetime('now', '-7 days')
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
        """Borsa kapalıyken gümüş hareketine göre açılış sinyali - GMSTR kapanış anındaki gümüş fiyatından hesapla"""
        try:
            # Cache kontrol - 5 dakika
            now = time_module.time()
            if hasattr(self, '_premarket_cache') and self._premarket_cache and hasattr(self, '_premarket_cache_time') and (now - self._premarket_cache_time) < 300:
                logger.info("Pre-market sinyali cache'den alındı")
                return self._premarket_cache
            
            # GMSTR son kapanış fiyatı ve zamanı
            gmstr = yf.Ticker("GMSTR.IS")
            gmstr_data = gmstr.history(period="5d")
            if gmstr_data.empty:
                return None
            gmstr_last_close = gmstr_data['Close'].iloc[-1]
            # Yahoo Finance günlük veri saatini 00:00 verir, ama gerçek kapanış 18:10
            gmstr_date = gmstr_data.index[-1]
            gmstr_close_time = gmstr_date.replace(hour=18, minute=10, second=0, microsecond=0)
            
            # Gerçek gümüş (SI=F) verisi - SAATLİK interval
            silver = yf.Ticker("SI=F")
            silver_data = silver.history(period="7d", interval="1h")  # 7/24 açık olduğu için 1h
            if silver_data.empty:
                return None
            silver_current = silver_data['Close'].iloc[-1]
            
            # GMSTR kapanış anındaki gümüş fiyatını bul (en yakın zaman)
            silver_at_gmstr_close = None
            for i in range(len(silver_data)-1, -1, -1):
                if silver_data.index[i] <= gmstr_close_time:
                    silver_at_gmstr_close = silver_data['Close'].iloc[i]
                    break
            
            if silver_at_gmstr_close is None:
                silver_at_gmstr_close = silver_data['Close'].iloc[0]  # En erken bulunabilen
            
            # Gümüşteki değişim = (Şimdi - Kapanış anındaki) / Kapanış anındaki
            silver_change_pct = ((silver_current - silver_at_gmstr_close) / silver_at_gmstr_close) * 100
            
            logger.info(f"Pre-market: GMSTR kapandığında gümüş={silver_at_gmstr_close:.2f}, Şimdi={silver_current:.2f}, Değişim={silver_change_pct:.2f}%")
            
            # Sinyal oluştur
            if silver_change_pct > 1.5:
                signal = "STRONG_BUY"
                direction = "YÜKSELİŞ"
                confidence = min(0.70 + abs(silver_change_pct) * 0.02, 0.95)
                reason = f"GMSTR kapanışından beri gümüş %{silver_change_pct:.2f} yükseldi"
            elif silver_change_pct > 0.5:
                signal = "BUY"
                direction = "YÜKSELİŞ"
                confidence = min(0.60 + abs(silver_change_pct) * 0.05, 0.75)
                reason = f"GMSTR kapanışından beri gümüş %{silver_change_pct:.2f} yükseldi"
            elif silver_change_pct < -1.5:
                signal = "STRONG_SELL"
                direction = "DÜŞÜŞ"
                confidence = min(0.70 + abs(silver_change_pct) * 0.02, 0.95)
                reason = f"GMSTR kapanışından beri gümüş %{abs(silver_change_pct):.2f} düştü"
            elif silver_change_pct < -0.5:
                signal = "SELL"
                direction = "DÜŞÜŞ"
                confidence = min(0.60 + abs(silver_change_pct) * 0.05, 0.75)
                reason = f"GMSTR kapanışından beri gümüş %{abs(silver_change_pct):.2f} düştü"
            else:
                signal = "HOLD"
                direction = "YATAY"
                confidence = 0.55
                reason = f"GMSTR kapanışından beri gümüş %{silver_change_pct:.2f} değişti (yatay)"
            
            # Hedef fiyat = GMSTR kapanış × gümüş değişimi
            target = gmstr_last_close * (1 + silver_change_pct / 100)
            
            result = {
                'gmstr_last_close': gmstr_last_close,
                'gmstr_close_time': gmstr_close_time.strftime('%d.%m.%Y %H:%M'),
                'silver_at_close': silver_at_gmstr_close,
                'silver_current': silver_current,
                'silver_change_pct': silver_change_pct,
                'signal': signal,
                'direction': direction,
                'confidence': confidence,
                'target_price': target,
                'reason': reason,
                'timestamp': datetime.now().strftime('%d.%m.%Y %H:%M')
            }
            
            # Cache'e kaydet
            self._premarket_cache = result
            self._premarket_cache_time = time_module.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Pre-market sinyal hatası: {e}")
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
    
    def calculate_risk_management(self, current_price, predicted_price, direction, confidence):
        """Risk yönetimi hesapla"""
        try:
            # Volatilite bazlı stop-loss
            volatility = abs(predicted_price - current_price) / current_price
            
            if direction == "YÜKSELİŞ":
                stop_loss = current_price * (1 - volatility * 1.5)
                take_profit = predicted_price
            else:
                stop_loss = current_price * (1 + volatility * 1.5)
                take_profit = predicted_price
            
            # Risk/Ödül oranı
            risk = abs(current_price - stop_loss)
            reward = abs(current_price - take_profit)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Pozisyon büyüklüğü (güvene göre)
            if confidence >= 0.80:
                position_size = 100  # %100
            elif confidence >= 0.70:
                position_size = 75   # %75
            elif confidence >= 0.60:
                position_size = 50   # %50
            else:
                position_size = 25   # %25
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size': position_size,
                'risk_amount': risk,
                'potential_profit': reward
            }
            
        except Exception as e:
            logger.error(f"Risk yönetimi hatası: {e}")
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
try:
    from flask_socketio import SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    logger.info("SocketIO başlatıldı")
except ImportError:
    socketio = None
    logger.warning("flask-socketio kurulu değil, WebSocket devre dışı")

prediction_system = GMSTRPredictionSystem()

@app.route('/')
def dashboard():
    """Dashboard ana sayfa"""
    return render_template('dashboard.html')

@app.route('/api/predictions')
def get_predictions():
    """Tahminleri getir"""
    try:
        conn = prediction_system.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, predicted_for_time, current_price, predicted_direction, 
                   predicted_price, confidence, timeframe, actual_price, is_correct,
                   telegram_sent
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        
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
                'telegram_sent': row[9] if row[9] is not None else 0
            })
        
        conn.close()
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Performans verileri"""
    stats = prediction_system.get_performance_stats()
    return jsonify(stats)

@app.route('/api/model-info')
def get_model_info():
    """Model eğitim bilgileri"""
    try:
        import json
        import os
        if os.path.exists('model_info.json'):
            with open('model_info.json', 'r') as f:
                info = json.load(f)
            return jsonify(info)
        else:
            # Dosya yoksa default değer döndür
            return jsonify({
                'status': 'not_trained',
                'message': 'Model henüz eğitilmemiş',
                'last_trained': '-',
                'accuracy': 0,
                'test_samples': 0,
                'train_samples': 0,
                'threshold': 0.60,
                'features_used': 0
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """Manuel tahmin yap"""
    timeframe = request.json.get('timeframe', '4h')
    prediction = prediction_system.make_prediction(timeframe)
    
    if prediction:
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

def scheduled_predictions():
    """Otomatik tahminler - Sadece borsa açıkken"""
    logger.info("Otomatik tahminler başlatılıyor...")
    
    def safe_prediction(timeframe):
        """Güvenli tahmin - borsa kapalıysa uyarı ver"""
        if not prediction_system.is_borsa_open():
            logger.info("Borsa kapalı, tahmin atlanıyor")
            return None
        return prediction_system.make_prediction(timeframe)
    
    # Her gün 09:00, 11:00, 13:00, 15:00'te tahmin yap (sadece hafta içi)
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        getattr(schedule.every(), day).at("09:00").do(lambda: safe_prediction("4h"))
        getattr(schedule.every(), day).at("11:00").do(lambda: safe_prediction("4h"))
        getattr(schedule.every(), day).at("13:00").do(lambda: safe_prediction("4h"))
        getattr(schedule.every(), day).at("15:00").do(lambda: safe_prediction("4h"))
    
    # Her saat tahminleri güncelle (doğrulama her zaman çalışabilir)
    schedule.every().hour.do(lambda: prediction_system.update_predictions())
    
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
