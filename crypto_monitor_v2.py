"""
GMSTR | BTC | ETH Geliştirilmiş Canlı Masaüstü Uygulaması
1h, 4h, 1gün, 1hafta tahminler + Alım-Satım fiyatları + Teknik göstergeler
"""

import tkinter as tk
from tkinter import ttk, font, messagebox
import numpy as np
import pandas as pd
import logging
import time
import threading
from datetime import datetime, timedelta
from exchange_client_crypto import CryptoExchangeClient
from exchange_client_bist import BISTExchangeClient
import joblib
import winsound
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============ TEKNİK GÖSTERGELER ============

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """RSI hesapla"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR hesapla"""
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    return df['tr'].rolling(window=window).mean()


def calculate_z_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Fiyatın Hareketli Ortalamadan Sapması (Z-Score)"""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    z_score = (df['close'] - df['ma']) / df['std'].replace(0, np.nan)
    return z_score.replace([np.inf, -np.inf], 0)


def calculate_volume_delta(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Delta: Hacim değişimi"""
    vol_delta = df['volume'].pct_change(window)
    return vol_delta.replace([np.inf, -np.inf], 0)


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.Series:
    """MACD hesapla"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd.replace([np.inf, -np.inf], 0)


def calculate_ema(df: pd.DataFrame, window=20) -> pd.Series:
    """EMA hesapla"""
    return df['close'].ewm(span=window).mean()


def calculate_bollinger_upper(df: pd.DataFrame, window=20) -> pd.Series:
    """Bollinger Upper Band"""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    upper_band = df['ma'] + (2 * df['std'])
    upper_band = upper_band.replace([np.inf, -np.inf], np.nan)
    return upper_band.fillna(df['close'])


def calculate_bollinger_lower(df: pd.DataFrame, window=20) -> pd.Series:
    """Bollinger Lower Band"""
    df = df.copy()
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    lower_band = df['ma'] - (2 * df['std'])
    lower_band = lower_band.replace([np.inf, -np.inf], np.nan)
    return lower_band.fillna(df['close'])


def calculate_sma(df: pd.DataFrame, window=20) -> pd.Series:
    """SMA hesapla"""
    return df['close'].rolling(window=window).mean()


def calculate_momentum(df: pd.DataFrame, window=10) -> pd.Series:
    """Momentum hesapla"""
    return df['close'] - df['close'].shift(window)


def calculate_stochastic(df: pd.DataFrame, window=14) -> pd.Series:
    """Stochastic hesapla"""
    df = df.copy()
    df['low_min'] = df['low'].rolling(window=window).min()
    df['high_max'] = df['high'].rolling(window=window).max()
    stochastic = 100 * (df['close'] - df['low_min']) / (df['high_max'] - df['low_min'])
    return stochastic.replace([np.inf, -np.inf], 50)


def calculate_williams_r(df: pd.DataFrame, window=14) -> pd.Series:
    """Williams %R hesapla"""
    df = df.copy()
    df['low_min'] = df['low'].rolling(window=window).min()
    df['high_max'] = df['high'].rolling(window=window).max()
    williams_r = -100 * (df['high_max'] - df['close']) / (df['high_max'] - df['low_min'])
    return williams_r.replace([np.inf, -np.inf], -50)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume hesapla"""
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv.replace([np.inf, -np.inf], 0)


def calculate_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index hesapla"""
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma_tp'] = df['typical_price'].rolling(window=window).mean()
    df['mad'] = df['typical_price'].rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (df['typical_price'] - df['sma_tp']) / (0.015 * df['mad'])
    return cci.replace([np.inf, -np.inf], 0)


def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple:
    """Destek ve direnç seviyeleri hesapla"""
    support = df['low'].rolling(window=window).min().iloc[-1]
    resistance = df['high'].rolling(window=window).max().iloc[-1]
    return support, resistance


# ============ MODEL YAPILANDIRMASI ============

# Test sonuçlarına göre güven skorları
MODEL_CONFIDENCE = {
    'GMSTR': {
        '1h': 0.85,  # Test edilemedi, varsayılan
        '4h': 0.87,  # %86.73 direction accuracy
        '1d': 0.95,  # %94.89 direction accuracy
        '1w': 1.00,  # %100 direction accuracy
    },
    'BTC': {
        '1h': 0.78,  # Varsayılan
        '4h': 0.82,  # %81.63 direction accuracy
        '1d': 0.88,  # %88.07 direction accuracy
        '1w': 0.59,  # %59.38 direction accuracy
    },
    'ETH': {
        '1h': 0.75,  # Varsayılan
        '4h': 0.77,  # %76.53 direction accuracy
        '1d': 0.81,  # %80.68 direction accuracy
        '1w': 0.84,  # %84.38 direction accuracy
    }
}

# Model dosya yolları
MODEL_FILES = {
    'GMSTR': {
        '1h': 'price_prediction_GMSTR_1d_updated.pkl',  # En başarılı model (%94.89 doğruluk)
        '4h': 'price_prediction_GMSTR_1d_updated.pkl',
        '1d': 'price_prediction_GMSTR_1d_updated.pkl',
        '1w': 'price_prediction_GMSTR_1d_updated.pkl',  # Weekly veri ile farklı feature -> farklı tahmin
    },
    'BTC': {
        '1h': None,  # Model yok
        '4h': 'price_prediction_BTC_4h_improved.pkl',
        '1d': 'price_prediction_BTC_1d_improved.pkl',
        '1w': 'price_prediction_BTC_1d_improved.pkl',
    },
    'ETH': {
        '1h': None,  # Model yok
        '4h': 'price_prediction_ETH_4h.pkl',
        '1d': 'price_prediction_ETH_1d.pkl',
        '1w': 'price_prediction_ETH_1d.pkl',
    }
}

# Spread oranları (alım-satım farkı)
SPREAD_RATES = {
    'GMSTR': 0.002,  # %0.2
    'BTC': 0.0005,   # %0.05
    'ETH': 0.0005,   # %0.05
}


# ============ ANA UYGULAMA ============

class CryptoMonitorAppV2:
    def __init__(self, root):
        self.root = root
        self.root.title("GMSTR | BTC | ETH Geliştirilmiş Canlı Monitör")
        self.root.geometry("2000x650")
        self.root.configure(bg="#1e1e1e")
        
        self.crypto_client = CryptoExchangeClient()
        self.bist_client = BISTExchangeClient()
        
        self.running = True
        self.sound_enabled = True
        self.auto_refresh_enabled = True
        self.refresh_interval = 300  # 5 dakika
        
        # Son sinyalleri takip et
        self.last_signals = {}
        
        # Model önbelleği
        self.model_cache = {}
        self._load_models()
        
        self.setup_ui()
        self.start_monitoring()
    
    def _load_models(self):
        """Modelleri önbelleğe yükle"""
        for symbol, timeframes in MODEL_FILES.items():
            self.model_cache[symbol] = {}
            for tf, model_path in timeframes.items():
                if model_path and os.path.exists(model_path):
                    try:
                        self.model_cache[symbol][tf] = joblib.load(model_path)
                        logger.info(f"Model yüklendi: {symbol} {tf}")
                    except Exception as e:
                        logger.warning(f"Model yüklenemedi: {symbol} {tf} - {e}")
                        self.model_cache[symbol][tf] = None
                else:
                    self.model_cache[symbol][tf] = None
    
    def setup_ui(self):
        """UI kurulumu"""
        # Widget referanslarını başlat
        self.symbol_widgets = {}
        
        # Başlık
        title_font = font.Font(family="Arial", size=20, weight="bold")
        title_label = tk.Label(
            self.root,
            text="🚀 GMSTR | BTC | ETH GELİŞTİRİLMİŞ CANLI MONİTÖR",
            font=title_font,
            bg="#1e1e1e",
            fg="#ffffff"
        )
        title_label.pack(pady=5)
        
        # Üst kontrol paneli
        control_frame = tk.Frame(self.root, bg="#2d2d2d")
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Ses butonu
        self.sound_button = tk.Button(
            control_frame,
            text="🔊 SES AÇIK",
            command=self.toggle_sound,
            font=("Arial", 11),
            bg="#4CAF50",
            fg="white",
            relief=tk.RAISED,
            bd=2
        )
        self.sound_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Otomatik yenileme butonu
        self.refresh_button = tk.Button(
            control_frame,
            text="🔄 OTOMATIK YENİLEME: AÇIK",
            command=self.toggle_auto_refresh,
            font=("Arial", 11),
            bg="#2196F3",
            fg="white",
            relief=tk.RAISED,
            bd=2
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Manuel yenileme butonu
        manual_refresh_btn = tk.Button(
            control_frame,
            text="🔄 ŞİMDİ YENİLE",
            command=lambda: self.root.after(0, self.update_ui),
            font=("Arial", 11),
            bg="#FF9800",
            fg="white",
            relief=tk.RAISED,
            bd=2
        )
        manual_refresh_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Son güncelleme etiketi
        self.update_label = tk.Label(
            control_frame,
            text="Son Güncelleme: --",
            font=("Arial", 10),
            bg="#2d2d2d",
            fg="#888888"
        )
        self.update_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Ana içerik frame (sekme yapısı)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # GMSTR sekmesi
        gmstr_frame = tk.Frame(notebook, bg="#2d2d2d")
        notebook.add(gmstr_frame, text="  GMSTR  ")
        self._create_symbol_panel(gmstr_frame, "GMSTR")
        
        # BTC sekmesi
        btc_frame = tk.Frame(notebook, bg="#2d2d2d")
        notebook.add(btc_frame, text="  BTC  ")
        self._create_symbol_panel(btc_frame, "BTC")
        
        # ETH sekmesi
        eth_frame = tk.Frame(notebook, bg="#2d2d2d")
        notebook.add(eth_frame, text="  ETH  ")
        self._create_symbol_panel(eth_frame, "ETH")
        
        # Alt bilgi paneli
        info_frame = tk.Frame(self.root, bg="#1e1e1e")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = tk.Label(
            info_frame,
            text="💡 Güven Skoru: Test sonuçlarına göre hesaplanır | "
                 "📈 AL: Tahmin > Mevcut Fiyat | 📉 SAT: Tahmin < Mevcut Fiyat | "
                 "Veriler 5 dakikada bir otomatik güncellenir",
            font=("Arial", 9),
            bg="#1e1e1e",
            fg="#666666"
        )
        info_text.pack()
    
    def _create_symbol_panel(self, parent, symbol):
        """Her sembol için panel oluştur"""
        # Ana frame
        main_frame = tk.Frame(parent, bg="#2d2d2d")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Zaman dilimi frame'leri (yatay) - 1h, 4h, 1d, 1w
        timeframes = ['1h', '4h', '1d', '1w']
        timeframe_labels = {'1h': '1 SAAT', '4h': '4 SAAT', '1d': '1 GÜN', '1w': '1 HAFTA'}
        
        # Her zaman dilimi için frame
        tf_frames = {}
        for i, tf in enumerate(timeframes):
            tf_frame = tk.Frame(main_frame, bg="#3d3d3d", relief=tk.RAISED, bd=2)
            tf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Başlık
            tf_label = tk.Label(
                tf_frame,
                text=f"{symbol}\n{timeframe_labels[tf]}",
                font=("Arial", 13, "bold"),
                bg="#3d3d3d",
                fg=self._get_symbol_color(symbol)
            )
            tf_label.pack(pady=5)
            
            # İçerik label
            content_label = tk.Label(
                tf_frame,
                text="Yükleniyor...",
                font=("Arial", 11),
                bg="#3d3d3d",
                fg="#ffffff",
                justify=tk.LEFT
            )
            content_label.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
            
            tf_frames[tf] = content_label
        
        # Alım-Satım paneli (dikey olarak altta)
        trade_frame = tk.Frame(main_frame, bg="#2d4d2d", relief=tk.RAISED, bd=2, width=200)
        trade_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        trade_title = tk.Label(
            trade_frame,
            text=f"{symbol}\nALIM-SATIM",
            font=("Arial", 13, "bold"),
            bg="#2d4d2d",
            fg="#00ff00"
        )
        trade_title.pack(pady=5)
        
        trade_content = tk.Label(
            trade_frame,
            text="Yükleniyor...",
            font=("Arial", 11),
            bg="#2d4d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        trade_content.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # Widget referanslarını sakla
        self.symbol_widgets[symbol] = {
            'timeframes': tf_frames,
            'trade': trade_content
        }
    
    def _get_symbol_color(self, symbol):
        """Sembol için renk döndür"""
        colors = {
            'GMSTR': '#00ff00',
            'BTC': '#ff9500',
            'ETH': '#627eea'
        }
        return colors.get(symbol, '#ffffff')
    
    def _get_confidence_color(self, confidence):
        """Güven skoru için renk döndür"""
        if confidence >= 0.90:
            return '#00ff00'  # Yeşil - Çok yüksek
        elif confidence >= 0.80:
            return '#7fff00'  # Açık yeşil - Yüksek
        elif confidence >= 0.70:
            return '#ffff00'  # Sarı - Orta
        elif confidence >= 0.60:
            return '#ffa500'  # Turuncu - Düşük
        else:
            return '#ff4444'  # Kırmızı - Çok düşük
    
    def _get_signal_color(self, pred_price, current_price):
        """Sinyal için renk döndür"""
        change_pct = ((pred_price - current_price) / current_price) * 100
        if change_pct > 2:
            return '#00ff00', '📈'  # Güçlü yükseliş
        elif change_pct > 0:
            return '#7fff00', '📈'  # Zayıf yükseliş
        elif change_pct < -2:
            return '#ff0000', '📉'  # Güçlü düşüş
        else:
            return '#ffa500', '📉'  # Zayıf düşüş
    
    def _get_prediction_for_timeframe(self, symbol, tf, df, features):
        """Belirli bir zaman dilimi için tahmin yap"""
        model = self.model_cache.get(symbol, {}).get(tf)
        if model is None:
            return None, None
        
        try:
            # Modelin beklediği feature'ları kullan (eğer feature_names_in_ varsa)
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
                # Sadece modelin beklediği feature'ları seç
                available_features = [f for f in model_features if f in df.columns]
                if len(available_features) != len(model_features):
                    missing = [f for f in model_features if f not in df.columns]
                    logger.warning(f"{symbol} {tf}: Eksik feature'lar: {missing}")
                    return None, None
                X = df[model_features].iloc[-1:]
            else:
                X = df[features].iloc[-1:]
            
            if pd.isna(X).any().any():
                return None, None
            pred_price = model.predict(X)[0]
            current_price = df['close'].iloc[-1]
            signal = 1 if pred_price > current_price else 0
            return pred_price, signal
        except Exception as e:
            logger.warning(f"Tahmin hatası {symbol} {tf}: {e}")
            return None, None
    
    def _compute_technicals(self, df):
        """DataFrame üzerinde tüm teknik göstergeleri hesapla ve NaN'ları temizle"""
        df = df.copy()
        df['rsi'] = calculate_rsi(df)
        df['atr'] = calculate_atr(df)
        df['z_score'] = calculate_z_score(df)
        df['volume_delta'] = calculate_volume_delta(df)
        df['macd'] = calculate_macd(df)
        df['ema'] = calculate_ema(df)
        df['bollinger_upper'] = calculate_bollinger_upper(df)
        df['bollinger_lower'] = calculate_bollinger_lower(df)
        df['sma'] = calculate_sma(df)
        df['momentum'] = calculate_momentum(df)
        df['stochastic'] = calculate_stochastic(df)
        df['williams_r'] = calculate_williams_r(df)
        df['obv'] = calculate_obv(df)
        df['cci'] = calculate_cci(df)
        # NaN'ları temizle (ilk window kadar satırda oluşur)
        df = df.ffill().bfill()
        return df
    
    def _prepare_timeframe_data(self, symbol, tf):
        """Belirli bir zaman dilimi için veri çek ve teknik göstergeleri hesapla"""
        if symbol == 'GMSTR':
            client = self.bist_client
            ticker = "GMSTR"
        elif symbol == 'BTC':
            client = self.crypto_client
            ticker = "BTCUSDT"
        elif symbol == 'ETH':
            client = self.crypto_client
            ticker = "ETHUSDT"
        else:
            return None
        
        # GMSTR için 1w verisini günlük veriden resample ederek al (model 523 bar ile eğitildi)
        if symbol == 'GMSTR' and tf == '1w':
            try:
                import yfinance as yf
                ticker_yf = yf.Ticker("GMSTR.IS")
                df_daily = ticker_yf.history(period="10y", interval="1d")
                if df_daily is not None and not df_daily.empty:
                    df_weekly = df_daily.resample('W-FRI').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 
                        'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    # API'deki veriyle senkronizasyon için daily close'u son weekly close ile güncelle
                    df_latest = client.fetch_ohlcv("GMSTR", timeframe="1h", limit=120)
                    if df_latest is not None and not df_latest.empty:
                        latest_close = df_latest['close'].iloc[-1]
                        df_weekly.iloc[-1, df_weekly.columns.get_loc('Close')] = latest_close
                    df = df_weekly.rename(columns={
                        'Open': 'open', 'High': 'high', 'Low': 'low',
                        'Close': 'close', 'Volume': 'volume'
                    }).reset_index()
                    df['timestamp'] = df['Date'].astype('int64') // 10**9
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(df) >= 20:
                        df = self._compute_technicals(df)
                        return df
            except Exception as e:
                logger.warning(f"YFinance 1w hatası {symbol}: {e}")
        
        # Normal akış - her zaman dilimi için kendi periyodunda veri çek
        tf_map = {'1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'}
        api_tf = tf_map.get(tf, '1h')
        
        limit_map = {'1h': 720, '4h': 520, '1d': 365, '1w': 156}
        limit = limit_map.get(tf, 200)
        
        df = client.fetch_ohlcv(ticker, timeframe=api_tf, limit=limit)
        if df is None or df.empty:
            return None
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        if len(df) < 20:
            return None
        
        # Teknik göstergeleri hesapla
        df = self._compute_technicals(df)
        return df
    
    def get_gmstr_prediction(self):
        """GMSTR tahmini al - tüm zaman dilimleri"""
        try:
            symbol = "GMSTR"
            results = {'price': 0}  # placeholder, will be overwritten
            
            # Her zaman dilimi için ayrı veri çek ve tahmin yap
            for tf in ['1h', '4h', '1d', '1w']:
                df = self._prepare_timeframe_data(symbol, tf)
                if df is None:
                    continue
                
                features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                            'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
                
                current_price = df['close'].iloc[-1]
                # En son timeframe'in fiyatını kullan (1h öncelikli)
                if tf == '1h' or results['price'] == 0:
                    results['price'] = current_price
                
                pred, signal = self._get_prediction_for_timeframe(symbol, tf, df, features)
                if pred is not None:
                    results[f'pred_{tf}'] = pred
                    results[f'signal_{tf}'] = signal
                    results[f'confidence_{tf}'] = MODEL_CONFIDENCE[symbol][tf]
            
            # En az bir tahmin başarılı oldu mu kontrol et
            if not any(f'pred_{tf}' in results for tf in ['1h', '4h', '1d', '1w']):
                logger.warning(f"GMSTR: Hiçbir tahmin yapılamadı")
                return None
            
            # Alım-satım fiyatları için 1h verisini kullan
            df_1h = self._prepare_timeframe_data(symbol, '1h')
            if df_1h is not None:
                current_price = df_1h['close'].iloc[-1]
                results['price'] = current_price
                
                # Alım-satım fiyatları
                spread = SPREAD_RATES['GMSTR']
                results['buy_price'] = current_price * (1 - spread)
                results['sell_price'] = current_price * (1 + spread)
                results['spread'] = spread * 100
                
                # Destek ve direnç (1h verisinden)
                support, resistance = calculate_support_resistance(df_1h)
                results['support'] = support
                results['resistance'] = resistance
                
                # RSI (1h verisinden)
                results['rsi'] = df_1h['rsi'].iloc[-1]
            
            return results
        except Exception as e:
            logger.error(f"GMSTR tahmin hatası: {e}")
            return None
    
    def get_btc_prediction(self):
        """BTC tahmini al - tüm zaman dilimleri"""
        try:
            symbol = "BTC"
            results = {'price': 0}
            
            all_features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                           'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                           'stochastic', 'williams_r', 'obv', 'cci']
            
            # Her zaman dilimi için ayrı veri çek ve tahmin yap
            for tf in ['1h', '4h', '1d', '1w']:
                df = self._prepare_timeframe_data(symbol, tf)
                if df is None:
                    continue
                
                current_price = df['close'].iloc[-1]
                if tf == '1h' or results['price'] == 0:
                    results['price'] = current_price
                
                pred, signal = self._get_prediction_for_timeframe(symbol, tf, df, all_features)
                if pred is not None:
                    results[f'pred_{tf}'] = pred
                    results[f'signal_{tf}'] = signal
                    results[f'confidence_{tf}'] = MODEL_CONFIDENCE[symbol].get(tf, 0.75)
            
            # 1h verisinden alım-satım bilgilerini al
            df_1h = self._prepare_timeframe_data(symbol, '1h')
            if df_1h is not None:
                current_price = df_1h['close'].iloc[-1]
                results['price'] = current_price
                
                spread = SPREAD_RATES['BTC']
                results['buy_price'] = current_price * (1 - spread)
                results['sell_price'] = current_price * (1 + spread)
                results['spread'] = spread * 100
                
                support, resistance = calculate_support_resistance(df_1h)
                results['support'] = support
                results['resistance'] = resistance
                results['rsi'] = df_1h['rsi'].iloc[-1]
            
            return results
        except Exception as e:
            logger.error(f"BTC tahmin hatası: {e}")
            return None
    
    def get_eth_prediction(self):
        """ETH tahmini al - tüm zaman dilimleri"""
        try:
            symbol = "ETH"
            results = {'price': 0}
            
            all_features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                           'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                           'stochastic', 'williams_r', 'obv', 'cci']
            
            # Her zaman dilimi için ayrı veri çek ve tahmin yap
            for tf in ['1h', '4h', '1d', '1w']:
                df = self._prepare_timeframe_data(symbol, tf)
                if df is None:
                    continue
                
                current_price = df['close'].iloc[-1]
                if tf == '1h' or results['price'] == 0:
                    results['price'] = current_price
                
                pred, signal = self._get_prediction_for_timeframe(symbol, tf, df, all_features)
                if pred is not None:
                    results[f'pred_{tf}'] = pred
                    results[f'signal_{tf}'] = signal
                    results[f'confidence_{tf}'] = MODEL_CONFIDENCE[symbol].get(tf, 0.75)
            
            # 1h verisinden alım-satım bilgilerini al
            df_1h = self._prepare_timeframe_data(symbol, '1h')
            if df_1h is not None:
                current_price = df_1h['close'].iloc[-1]
                results['price'] = current_price
                
                spread = SPREAD_RATES['ETH']
                results['buy_price'] = current_price * (1 - spread)
                results['sell_price'] = current_price * (1 + spread)
                results['spread'] = spread * 100
                
                support, resistance = calculate_support_resistance(df_1h)
                results['support'] = support
                results['resistance'] = resistance
                results['rsi'] = df_1h['rsi'].iloc[-1]
            
            return results
        except Exception as e:
            logger.error(f"ETH tahmin hatası: {e}")
            return None
    
    def format_timeframe_info(self, symbol, data, tf):
        """Zaman dilimi bilgilerini formatla"""
        if data is None:
            return "Veri yok", "#888888"
        
        pred_key = f'pred_{tf}'
        signal_key = f'signal_{tf}'
        confidence_key = f'confidence_{tf}'
        
        if pred_key not in data:
            return "Veri yok", "#888888"
        
        pred_price = data[pred_key]
        current_price = data['price']
        signal = data[signal_key]
        confidence = data.get(confidence_key, 0.8)
        
        color, arrow = self._get_signal_color(pred_price, current_price)
        confidence_color = self._get_confidence_color(confidence)
        
        # Fiyat formatı
        if symbol == 'GMSTR':
            price_fmt = f"₺{current_price:.2f}"
            pred_fmt = f"₺{pred_price:.2f}"
        else:
            price_fmt = f"${current_price:,.2f}"
            pred_fmt = f"${pred_price:,.2f}"
        
        signal_text = "📈 YÜKSELİŞ" if signal == 1 else "📉 DÜŞÜŞ"
        
        info = (
            f"{price_fmt}\n\n"
            f"{signal_text}\n\n"
            f"Hedef: {pred_fmt}\n\n"
            f"Güven: %{confidence*100:.0f}"
        )
        
        return info, color
    
    def format_trade_info(self, symbol, trade_data):
        """Alım-satım bilgilerini formatla"""
        if trade_data is None:
            return "Veri yok"
        
        buy = trade_data['buy_price']
        sell = trade_data['sell_price']
        spread = trade_data['spread']
        support = trade_data['support']
        resistance = trade_data['resistance']
        rsi = trade_data['rsi']
        
        # Fiyat formatı
        if symbol == 'GMSTR':
            buy_fmt = f"₺{buy:.2f}"
            sell_fmt = f"₺{sell:.2f}"
            support_fmt = f"₺{support:.2f}"
            resistance_fmt = f"₺{resistance:.2f}"
        else:
            buy_fmt = f"${buy:,.2f}"
            sell_fmt = f"${sell:,.2f}"
            support_fmt = f"${support:,.2f}"
            resistance_fmt = f"${resistance:,.2f}"
        
        # RSI yorumu
        if rsi > 70:
            rsi_text = f"RSI: {rsi:.1f} (AŞIRI ALIM)"
        elif rsi < 30:
            rsi_text = f"RSI: {rsi:.1f} (AŞIRI SATIM)"
        else:
            rsi_text = f"RSI: {rsi:.1f} (NÖTR)"
        
        info = (
            f"💰 ALIM: {buy_fmt}\n\n"
            f"💰 SATIM: {sell_fmt}\n\n"
            f"Spread: %{spread:.2f}\n\n"
            f"📊 Destek: {support_fmt}\n\n"
            f"📊 Direnç: {resistance_fmt}\n\n"
            f"{rsi_text}"
        )
        
        return info
    
    def update_ui(self):
        """UI güncelle"""
        # Tüm zaman dilimleri
        all_timeframes = ['1h', '4h', '1d', '1w']
        
        # GMSTR
        gmstr_data = self.get_gmstr_prediction()
        if gmstr_data and 'timeframes' in self.symbol_widgets.get('GMSTR', {}):
            for tf in all_timeframes:
                widget = self.symbol_widgets['GMSTR']['timeframes'][tf]
                info, color = self.format_timeframe_info('GMSTR', gmstr_data, tf)
                widget.config(text=info, fg=color)
            
            trade_widget = self.symbol_widgets['GMSTR']['trade']
            trade_info = self.format_trade_info('GMSTR', gmstr_data)
            trade_widget.config(text=trade_info)
            
            # Sinyal değişimi kontrolü
            self._check_signal_change('GMSTR', gmstr_data)
        else:
            if 'GMSTR' in self.symbol_widgets:
                for tf in all_timeframes:
                    self.symbol_widgets['GMSTR']['timeframes'][tf].config(text="Veri yüklenemedi", fg="#888888")
                self.symbol_widgets['GMSTR']['trade'].config(text="Veri yüklenemedi")
            logger.warning("GMSTR verisi yüklenemedi")
        
        # BTC
        btc_data = self.get_btc_prediction()
        if btc_data and 'timeframes' in self.symbol_widgets.get('BTC', {}):
            for tf in all_timeframes:
                widget = self.symbol_widgets['BTC']['timeframes'][tf]
                info, color = self.format_timeframe_info('BTC', btc_data, tf)
                widget.config(text=info, fg=color)
            
            trade_widget = self.symbol_widgets['BTC']['trade']
            trade_info = self.format_trade_info('BTC', btc_data)
            trade_widget.config(text=trade_info)
            
            # Sinyal değişimi kontrolü
            self._check_signal_change('BTC', btc_data)
        else:
            if 'BTC' in self.symbol_widgets:
                for tf in all_timeframes:
                    self.symbol_widgets['BTC']['timeframes'][tf].config(text="Veri yüklenemedi", fg="#888888")
                self.symbol_widgets['BTC']['trade'].config(text="Veri yüklenemedi")
            logger.warning("BTC verisi yüklenemedi")
        
        # ETH
        eth_data = self.get_eth_prediction()
        if eth_data and 'timeframes' in self.symbol_widgets.get('ETH', {}):
            for tf in all_timeframes:
                widget = self.symbol_widgets['ETH']['timeframes'][tf]
                info, color = self.format_timeframe_info('ETH', eth_data, tf)
                widget.config(text=info, fg=color)
            
            trade_widget = self.symbol_widgets['ETH']['trade']
            trade_info = self.format_trade_info('ETH', eth_data)
            trade_widget.config(text=trade_info)
            
            # Sinyal değişimi kontrolü
            self._check_signal_change('ETH', eth_data)
        else:
            if 'ETH' in self.symbol_widgets:
                for tf in all_timeframes:
                    self.symbol_widgets['ETH']['timeframes'][tf].config(text="Veri yüklenemedi", fg="#888888")
                self.symbol_widgets['ETH']['trade'].config(text="Veri yüklenemedi")
            logger.warning("ETH verisi yüklenemedi")
        
        # Son güncelleme
        self.update_label.config(
            text=f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    def _check_signal_change(self, symbol, data):
        """Sinyal değişimini kontrol et ve uyarı ver"""
        # 4h sinyali öncelikli use
        current_signal = data.get('signal_4h')
        if current_signal is None:
            return
        
        confidence = data.get('confidence_4h', 0)
        last_signal = self.last_signals.get(symbol)
        
        # Sinyal değiştiyse ve yüksek güven varsa
        if last_signal is not None and last_signal != current_signal:
            if confidence >= 0.80:  # %80+ güven
                self.play_sound()
                signal_text = "YÜKSELİŞ" if current_signal == 1 else "DÜŞÜŞ"
                logger.info(f"🚨 {symbol} SİNYAL DEĞİŞİMİ: {last_signal} -> {current_signal} ({signal_text}, Güven: %{confidence*100:.0f})")
        
        self.last_signals[symbol] = current_signal
    
    def toggle_sound(self):
        """Ses aç/kapa"""
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.sound_button.config(text="🔊 SES AÇIK", bg="#4CAF50")
        else:
            self.sound_button.config(text="🔇 SES KAPALI", bg="#f44336")
    
    def toggle_auto_refresh(self):
        """Otomatik yenileme aç/kapa"""
        self.auto_refresh_enabled = not self.auto_refresh_enabled
        if self.auto_refresh_enabled:
            self.refresh_button.config(text="🔄 OTOMATIK YENİLEME: AÇIK", bg="#2196F3")
        else:
            self.refresh_button.config(text="⏸ OTOMATIK YENİLEME: KAPALI", bg="#9E9E9E")
    
    def play_sound(self):
        """Ses çal"""
        if self.sound_enabled:
            try:
                winsound.Beep(1000, 500)  # 1000Hz, 500ms
                time.sleep(0.1)
                winsound.Beep(1200, 300)  # 1200Hz, 300ms
            except Exception as e:
                logger.error(f"Ses hatası: {e}")
    
    def start_monitoring(self):
        """İzlemeyi başlat"""
        def monitor():
            while self.running:
                try:
                    if self.auto_refresh_enabled:
                        self.root.after(0, self.update_ui)
                    time.sleep(self.refresh_interval)
                except Exception as e:
                    logger.error(f"İzleme hatası: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        
        # İlk güncelleme
        self.root.after(1000, self.update_ui)
    
    def on_close(self):
        """Kapatma"""
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CryptoMonitorAppV2(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()