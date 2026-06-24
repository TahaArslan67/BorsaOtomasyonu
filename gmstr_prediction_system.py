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
import time
import logging
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
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
        self.db_path = 'gmstr_predictions.db'
        self.model_path = 'gmstr_prediction_model.pkl'
        self.init_database()
        self.model = None
        self.features = []
        
        # Telegram Bot Config
        self.telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
        
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
        
    def init_database(self):
        """Veritabanını başlat"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tahminler tablosu
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
        
        conn.commit()
        conn.close()
        logger.info("Veritabanı başlatıldı")
    
    def fetch_gmstr_data(self, period="1y"):
        """GMSTR verilerini çek"""
        try:
            # Yahoo Finance'den GMSTR verisi
            ticker = yf.Ticker("GMSTR.IS")
            data = ticker.history(period=period, interval="1h")
            
            if data is None:
                logger.error("GMSTR verisi None döndü")
                return None
            
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
                    return None
            
            if data.empty:
                logger.error("GMSTR verisi çekilemedi")
                return None
                
            return data
        except Exception as e:
            logger.error(f"GMSTR veri çekme hatası: {e}")
            return None
    
    def fetch_market_data(self):
        """Piyasa verilerini çek"""
        try:
            # BIST 100
            bist100 = yf.Ticker("XU100.IS").history(period="1y", interval="1h")
            
            # USD/TRY
            usd_try = yf.Ticker("USDTRY=X").history(period="1y", interval="1h")
            
            # Gümüş fiyatı (alternatif semboller)
            silver = None
            for symbol in ["SILVER", "SI=F", "XAGUSD", "GC=F"]:  # Gümüş veya altın
                try:
                    silver = yf.Ticker(symbol).history(period="1y", interval="1h")
                    if not silver.empty:
                        logger.info(f"Alternatif sembol bulundu: {symbol}")
                        break
                except:
                    continue
            
            if silver is None or silver.empty:
                logger.warning("Gümüş verisi bulunamadı, alternatif kullanılacak")
                # Sahte gümüş verisi oluştur (USD/TRY bazında)
                if not usd_try.empty:
                    silver = usd_try.copy()
                    silver['Close'] = silver['Close'] * 0.05  # Yaklaşık gümüş fiyatı
                    logger.info("USD/TRY bazında sahte gümüş verisi oluşturuldu")
                else:
                    silver = None
            
            return {
                'bist100': bist100,
                'usd_try': usd_try,
                'silver': silver
            }
        except Exception as e:
            logger.error(f"Piyasa veri çekme hatası: {e}")
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
            
            # Özellik isimlerini önceden tanımla
            feature_names = []
            for indicator in ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 
                             'atr', 'stoch_k', 'stoch_d', 'williams_r', 'ema_20', 
                             'sma_50', 'momentum', 'volume_delta', 'obv', 'cci', 'z_score']:
                feature_names.append(f'gmstr_{indicator}')
            
            for indicator in ['rsi', 'macd', 'ema_20', 'sma_50']:
                feature_names.append(f'bist_{indicator}')
            
            feature_names.extend(['price_change_1h', 'price_change_4h', 'price_change_24h', 'volume_change'])
            
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
                
                # Piyasa göstergeleri
                for indicator in ['rsi', 'macd', 'ema_20', 'sma_50']:
                    val = 0.0
                    if bist_indicators and indicator in bist_indicators and i < len(bist_indicators[indicator]):
                        raw_val = bist_indicators[indicator].iloc[i]
                        val = safe_float(raw_val)
                    row.append(val)
                
                # Fiyat değişimleri
                if i >= 1:
                    price_change_1h = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-1]) / gmstr_data['Close'].iloc[i-1])
                else:
                    price_change_1h = 0.0
                row.append(price_change_1h)
                
                if i >= 4:
                    price_change_4h = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-4]) / gmstr_data['Close'].iloc[i-4])
                else:
                    price_change_4h = 0.0
                row.append(price_change_4h)
                
                if i >= 24:
                    price_change_24h = safe_float((gmstr_data['Close'].iloc[i] - gmstr_data['Close'].iloc[i-24]) / gmstr_data['Close'].iloc[i-24])
                else:
                    price_change_24h = 0.0
                row.append(price_change_24h)
                
                # Hacim değişimi
                if i >= 1:
                    volume_change = safe_float((gmstr_data['Volume'].iloc[i] - gmstr_data['Volume'].iloc[i-1]) / gmstr_data['Volume'].iloc[i-1])
                else:
                    volume_change = 0.0
                row.append(volume_change)
                
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
    
    def create_labels(self, data, timeframe_hours=4):
        """Etiketler oluştur (yön tahmini)"""
        labels = []
        
        for i in range(20, len(data) - timeframe_hours):
            current_price = data['Close'].iloc[i]
            future_price = data['Close'].iloc[i + timeframe_hours]
            
            if future_price > current_price * 1.005:  # %0.5'tan fazla artış
                labels.append(1)  # Yükseliş
            elif future_price < current_price * 0.995:  # %0.5'tan fazla düşüş
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
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Model seçimi ve eğitimi
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                avg_score = cv_scores.mean()
                
                logger.info(f"{name} CV Score: {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # En iyi modeli eğit
            best_model.fit(X_train, y_train)
            
            # Test performansı
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            if accuracy >= 0.65:
                logger.info(f"Model %65 başarı oranına ulaştı: {accuracy:.4f}")
                self.model = best_model
                joblib.dump(best_model, self.model_path)
                
                # Özellik isimlerini kaydet
                with open('feature_names.txt', 'w') as f:
                    f.write(','.join(self.features))
                
                return True
            else:
                logger.warning(f"Model %65 başarı oranına ulaşamadı: {accuracy:.4f}")
                return False
                
        except Exception as e:
            logger.error(f"Model eğitimi hatası: {e}")
            return False
    
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
            
            # Güncel verileri çek
            gmstr_data = self.fetch_gmstr_data(period="30d")
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (timestamp, predicted_for_time, current_price, predicted_direction, 
                predicted_price, confidence, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), predicted_for_time, current_price, direction, target_price, confidence, timeframe))
            
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE predictions SET telegram_sent = ? WHERE id = ?', (status, pred_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Telegram status güncelleme hatası: {e}")
    
    def update_predictions(self):
        """Tahminleri güncelle (doğruluk kontrolü) - Otomatik"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Doğrulanmamış tahminleri al (zamanı geçmiş olanları)
            cursor.execute('''
                SELECT id, timestamp, predicted_for_time, current_price, predicted_price, predicted_direction, confidence
                FROM predictions
                WHERE actual_price IS NULL
                AND (predicted_for_time < datetime('now') OR datetime(timestamp) < datetime('now', '-6 hours'))
            ''')
            
            predictions = cursor.fetchall()
            
            if len(predictions) == 0:
                return
            
            # O anki fiyatı al
            gmstr_data = self.fetch_gmstr_data(period="1d")
            if gmstr_data is None:
                logger.error("GMSTR verisi çekilemedi, tahminler güncellenemiyor")
                return
                
            actual_price = gmstr_data['Close'].iloc[-1]
            
            correct_count = 0
            total_count = len(predictions)
            
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
                cursor.execute('''
                    UPDATE predictions
                    SET actual_price = ?, actual_direction = ?, is_correct = ?
                    WHERE id = ?
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Son 7 günün performansı
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(is_correct) as correct,
                    COUNT(*) * 100.0 / NULLIF(SUM(is_correct), 0) as accuracy
                FROM predictions
                WHERE is_correct IS NOT NULL
                AND datetime(created_at) > datetime('now', '-7 days')
            ''')
            
            stats = cursor.fetchone()
            
            conn.close()
            
            if stats and stats[0] > 0:
                return {
                    'total_predictions': stats[0],
                    'correct_predictions': stats[1],
                    'accuracy': (stats[1] / stats[0]) * 100
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Performans istatistikleri hatası: {e}")
            return None

# Flask API
app = Flask(__name__)
prediction_system = GMSTRPredictionSystem()

@app.route('/')
def dashboard():
    """Dashboard ana sayfa"""
    return render_template('dashboard.html')

@app.route('/api/predictions')
def get_predictions():
    """Tahminleri getir"""
    try:
        conn = sqlite3.connect('gmstr_predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, current_price, predicted_direction, predicted_price, 
                   confidence, timeframe, actual_price, is_correct
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        
        predictions = []
        for row in cursor.fetchall():
            predictions.append({
                'timestamp': row[0],
                'predicted_for_time': row[1] if len(row) > 8 else None,
                'current_price': row[2] if len(row) > 8 else row[1],
                'predicted_direction': row[3] if len(row) > 8 else row[2],
                'predicted_price': row[4] if len(row) > 8 else row[3],
                'confidence': row[5] if len(row) > 8 else row[4],
                'timeframe': row[6] if len(row) > 8 else row[5],
                'actual_price': row[7] if len(row) > 8 else row[6],
                'is_correct': row[8] if len(row) > 8 else row[7],
                'telegram_sent': row[10] if len(row) > 10 else 0
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

def scheduled_predictions():
    """Otomatik tahminler"""
    logger.info("Otomatik tahminler başlatılıyor...")
    
    # Her gün 09:00 ve 15:00'te tahmin yap
    schedule.every().day.at("09:00").do(lambda: prediction_system.make_prediction("4h"))
    schedule.every().day.at("15:00").do(lambda: prediction_system.make_prediction("4h"))
    
    # Her saat tahminleri güncelle
    schedule.every().hour.do(lambda: prediction_system.update_predictions())
    
    while True:
        schedule.run_pending()
        time.sleep(60)

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
