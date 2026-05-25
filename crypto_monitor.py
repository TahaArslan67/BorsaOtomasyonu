"""
GMSTR, BTC, ETH Canlı Masaüstü Uygulaması
Monitör Uygulaması - Yatay Düzen
"""

import tkinter as tk
from tkinter import ttk, font
import numpy as np
import pandas as pd
import logging
import time
import threading
from datetime import datetime
from exchange_client_crypto import CryptoExchangeClient
from exchange_client_bist import BISTExchangeClient
import joblib
import winsound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


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


class CryptoMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GMSTR | BTC | ETH Canlı Monitör")
        self.root.geometry("1600x400")
        self.root.configure(bg="#1e1e1e")
        
        self.crypto_client = CryptoExchangeClient()
        self.bist_client = BISTExchangeClient()
        
        self.running = True
        self.sound_enabled = True
        
        # Son sinyalleri takip et
        self.last_btc_signal = None
        self.last_eth_signal = None
        
        self.setup_ui()
        self.start_monitoring()
    
    def setup_ui(self):
        """UI kurulumu"""
        # Başlık
        title_font = font.Font(family="Arial", size=24, weight="bold")
        title_label = tk.Label(
            self.root,
            text="GMSTR | BTC | ETH CANLI MONİTÖR",
            font=title_font,
            bg="#1e1e1e",
            fg="#ffffff"
        )
        title_label.pack(pady=10)
        
        # Ses butonu
        self.sound_button = tk.Button(
            self.root,
            text="🔊 SES AÇIK",
            command=self.toggle_sound,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            relief=tk.RAISED,
            bd=3
        )
        self.sound_button.pack(pady=5)
        
        # Yatay frame
        horizontal_frame = tk.Frame(self.root, bg="#1e1e1e")
        horizontal_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # GMSTR 4h Frame
        self.gmstr_4h_frame = tk.Frame(horizontal_frame, bg="#2d2d2d", relief=tk.RAISED, bd=3)
        self.gmstr_4h_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        gmstr_4h_title = tk.Label(
            self.gmstr_4h_frame,
            text="GMSTR\n4 SAAT",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="#00ff00"
        )
        gmstr_4h_title.pack(pady=10)
        
        self.gmstr_4h_info = tk.Label(
            self.gmstr_4h_frame,
            text="Yükleniyor...",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        self.gmstr_4h_info.pack(pady=10, padx=10)
        
        # GMSTR 1gün Frame
        self.gmstr_1d_frame = tk.Frame(horizontal_frame, bg="#2d2d2d", relief=tk.RAISED, bd=3)
        self.gmstr_1d_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        gmstr_1d_title = tk.Label(
            self.gmstr_1d_frame,
            text="GMSTR\n1 GÜN",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="#00ff00"
        )
        gmstr_1d_title.pack(pady=10)
        
        self.gmstr_1d_info = tk.Label(
            self.gmstr_1d_frame,
            text="Yükleniyor...",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        self.gmstr_1d_info.pack(pady=10, padx=10)
        
        # BTC 4h Frame
        self.btc_4h_frame = tk.Frame(horizontal_frame, bg="#2d2d2d", relief=tk.RAISED, bd=3)
        self.btc_4h_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        btc_4h_title = tk.Label(
            self.btc_4h_frame,
            text="BTC\n4 SAAT",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="#ff9500"
        )
        btc_4h_title.pack(pady=10)
        
        self.btc_4h_info = tk.Label(
            self.btc_4h_frame,
            text="Yükleniyor...",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        self.btc_4h_info.pack(pady=10, padx=10)
        
        # BTC 1gün Frame
        self.btc_1d_frame = tk.Frame(horizontal_frame, bg="#2d2d2d", relief=tk.RAISED, bd=3)
        self.btc_1d_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        btc_1d_title = tk.Label(
            self.btc_1d_frame,
            text="BTC\n1 GÜN",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="#ff9500"
        )
        btc_1d_title.pack(pady=10)
        
        self.btc_1d_info = tk.Label(
            self.btc_1d_frame,
            text="Yükleniyor...",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        self.btc_1d_info.pack(pady=10, padx=10)
        
        # ETH 4h Frame
        self.eth_4h_frame = tk.Frame(horizontal_frame, bg="#2d2d2d", relief=tk.RAISED, bd=3)
        self.eth_4h_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        eth_4h_title = tk.Label(
            self.eth_4h_frame,
            text="ETH\n4 SAAT",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="#627eea"
        )
        eth_4h_title.pack(pady=10)
        
        self.eth_4h_info = tk.Label(
            self.eth_4h_frame,
            text="Yükleniyor...",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        self.eth_4h_info.pack(pady=10, padx=10)
        
        # ETH 1gün Frame
        self.eth_1d_frame = tk.Frame(horizontal_frame, bg="#2d2d2d", relief=tk.RAISED, bd=3)
        self.eth_1d_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        eth_1d_title = tk.Label(
            self.eth_1d_frame,
            text="ETH\n1 GÜN",
            font=("Arial", 14, "bold"),
            bg="#2d2d2d",
            fg="#627eea"
        )
        eth_1d_title.pack(pady=10)
        
        self.eth_1d_info = tk.Label(
            self.eth_1d_frame,
            text="Yükleniyor...",
            font=("Arial", 12),
            bg="#2d2d2d",
            fg="#ffffff",
            justify=tk.LEFT
        )
        self.eth_1d_info.pack(pady=10, padx=10)
        
        # Son güncelleme
        self.update_label = tk.Label(
            self.root,
            text="Son Güncelleme: --",
            font=("Arial", 10),
            bg="#1e1e1e",
            fg="#888888"
        )
        self.update_label.pack(pady=5)
    
    def toggle_sound(self):
        """Ses aç/kapa"""
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.sound_button.config(text="🔊 SES AÇIK", bg="#4CAF50")
        else:
            self.sound_button.config(text="🔇 SES KAPALI", bg="#f44336")
    
    def play_sound(self):
        """Ses çal"""
        if self.sound_enabled:
            try:
                winsound.Beep(1000, 500)
            except Exception as e:
                logger.error(f"Ses hatası: {e}")
    
    def get_gmstr_prediction(self):
        """GMSTR tahmini al"""
        try:
            symbol = "GMSTR"
            df = self.bist_client.fetch_ohlcv(symbol, timeframe="1h", limit=720)
            
            if df is None or df.empty:
                return None
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
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
            
            features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                        'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
            
            X = df[features].iloc[-1:]
            current_price = df['close'].iloc[-1]
            
            model_price_4h = joblib.load(f"price_prediction_{symbol}_4h_updated.pkl")
            model_price_1d = joblib.load(f"price_prediction_{symbol}_1d_updated.pkl")
            
            pred_price_4h = model_price_4h.predict(X)[0]
            pred_price_1d = model_price_1d.predict(X)[0]
            
            signal_4h = 1 if pred_price_4h > current_price else 0
            signal_1d = 1 if pred_price_1d > current_price else 0
            
            return {
                'price': current_price,
                'pred_4h': pred_price_4h,
                'pred_1d': pred_price_1d,
                'signal_4h': signal_4h,
                'signal_1d': signal_1d
            }
        except Exception as e:
            logger.error(f"GMSTR tahmin hatası: {e}")
            return None
    
    def get_btc_prediction(self):
        """BTC tahmini al"""
        try:
            symbol = "BTCUSDT"
            df = self.crypto_client.fetch_ohlcv(symbol, timeframe="1h", limit=720)
            
            if df is None or df.empty:
                return None
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
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
            
            features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                        'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                        'stochastic', 'williams_r']
            
            X = df[features].iloc[-1:]
            current_price = df['close'].iloc[-1]
            
            model_price_4h = joblib.load(f"price_prediction_BTC_4h_improved.pkl")
            model_price_1d = joblib.load(f"price_prediction_BTC_1d_improved.pkl")
            
            pred_price_4h = model_price_4h.predict(X)[0]
            pred_price_1d = model_price_1d.predict(X)[0]
            
            signal_4h = 1 if pred_price_4h > current_price else 0
            signal_1d = 1 if pred_price_1d > current_price else 0
            
            return {
                'price': current_price,
                'pred_4h': pred_price_4h,
                'pred_1d': pred_price_1d,
                'signal_4h': signal_4h,
                'signal_1d': signal_1d
            }
        except Exception as e:
            logger.error(f"BTC tahmin hatası: {e}")
            return None
    
    def get_eth_prediction(self):
        """ETH tahmini al"""
        try:
            symbol = "ETHUSDT"
            df = self.crypto_client.fetch_ohlcv(symbol, timeframe="1h", limit=720)
            
            if df is None or df.empty:
                return None
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
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
            
            features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                        'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                        'stochastic', 'williams_r', 'obv', 'cci']
            
            X = df[features].iloc[-1:]
            current_price = df['close'].iloc[-1]
            
            model_price_4h = joblib.load(f"price_prediction_ETH_4h.pkl")
            model_price_1d = joblib.load(f"price_prediction_ETH_1d.pkl")
            
            pred_price_4h = model_price_4h.predict(X)[0]
            pred_price_1d = model_price_1d.predict(X)[0]
            
            signal_4h = 1 if pred_price_4h > current_price else 0
            signal_1d = 1 if pred_price_1d > current_price else 0
            
            return {
                'price': current_price,
                'pred_4h': pred_price_4h,
                'pred_1d': pred_price_1d,
                'signal_4h': signal_4h,
                'signal_1d': signal_1d
            }
        except Exception as e:
            logger.error(f"ETH tahmin hatası: {e}")
            return None
    
    def update_ui(self):
        """UI güncelle"""
        # GMSTR 4h
        gmstr_data = self.get_gmstr_prediction()
        if gmstr_data:
            signal_4h_text = "📈 YÜKSELİŞ" if gmstr_data['signal_4h'] == 1 else "📉 DÜŞÜŞ"
            signal_4h_color = "#00ff00" if gmstr_data['signal_4h'] == 1 else "#ff0000"
            
            self.gmstr_4h_info.config(
                text=f"₺{gmstr_data['price']:.2f}\n\n{signal_4h_text}\n\nHedef: ₺{gmstr_data['pred_4h']:.2f}",
                fg=signal_4h_color
            )
        else:
            self.gmstr_4h_info.config(text="Veri yüklenemedi", fg="#888888")
        
        # GMSTR 1gün
        if gmstr_data:
            signal_1d_text = "📈 YÜKSELİŞ" if gmstr_data['signal_1d'] == 1 else "📉 DÜŞÜŞ"
            signal_1d_color = "#00ff00" if gmstr_data['signal_1d'] == 1 else "#ff0000"
            
            self.gmstr_1d_info.config(
                text=f"₺{gmstr_data['price']:.2f}\n\n{signal_1d_text}\n\nHedef: ₺{gmstr_data['pred_1d']:.2f}",
                fg=signal_1d_color
            )
        else:
            self.gmstr_1d_info.config(text="Veri yüklenemedi", fg="#888888")
        
        # BTC 4h
        btc_data = self.get_btc_prediction()
        if btc_data:
            signal_4h_text = "📈 YÜKSELİŞ" if btc_data['signal_4h'] == 1 else "📉 DÜŞÜŞ"
            signal_4h_color = "#00ff00" if btc_data['signal_4h'] == 1 else "#ff0000"
            
            self.btc_4h_info.config(
                text=f"${btc_data['price']:.2f}\n\n{signal_4h_text}\n\nHedef: ${btc_data['pred_4h']:.2f}",
                fg=signal_4h_color
            )
        else:
            self.btc_4h_info.config(text="Veri yüklenemedi", fg="#888888")
        
        # BTC 1gün
        if btc_data:
            signal_1d_text = "📈 YÜKSELİŞ" if btc_data['signal_1d'] == 1 else "📉 DÜŞÜŞ"
            signal_1d_color = "#00ff00" if btc_data['signal_1d'] == 1 else "#ff0000"
            
            self.btc_1d_info.config(
                text=f"${btc_data['price']:.2f}\n\n{signal_1d_text}\n\nHedef: ${btc_data['pred_1d']:.2f}",
                fg=signal_1d_color
            )
        else:
            self.btc_1d_info.config(text="Veri yüklenemedi", fg="#888888")
        
        # ETH 4h
        eth_data = self.get_eth_prediction()
        if eth_data:
            signal_4h_text = "📈 YÜKSELİŞ" if eth_data['signal_4h'] == 1 else "📉 DÜŞÜŞ"
            signal_4h_color = "#00ff00" if eth_data['signal_4h'] == 1 else "#ff0000"
            
            self.eth_4h_info.config(
                text=f"${eth_data['price']:.2f}\n\n{signal_4h_text}\n\nHedef: ${eth_data['pred_4h']:.2f}",
                fg=signal_4h_color
            )
        else:
            self.eth_4h_info.config(text="Veri yüklenemedi", fg="#888888")
        
        # ETH 1gün
        if eth_data:
            signal_1d_text = "📈 YÜKSELİŞ" if eth_data['signal_1d'] == 1 else "📉 DÜŞÜŞ"
            signal_1d_color = "#00ff00" if eth_data['signal_1d'] == 1 else "#ff0000"
            
            self.eth_1d_info.config(
                text=f"${eth_data['price']:.2f}\n\n{signal_1d_text}\n\nHedef: ${eth_data['pred_1d']:.2f}",
                fg=signal_1d_color
            )
        else:
            self.eth_1d_info.config(text="Veri yüklenemedi", fg="#888888")
        
        # BTC sinyal değişimi kontrol et
        if btc_data:
            current_btc_signal = btc_data['signal_4h']  # 4h sinyalini kullan
            if self.last_btc_signal is not None and self.last_btc_signal == 0 and current_btc_signal == 1:
                self.play_sound()
            self.last_btc_signal = current_btc_signal
        
        # ETH sinyal değişimi kontrol et
        if eth_data:
            current_eth_signal = eth_data['signal_4h']  # 4h sinyalini kullan
            if self.last_eth_signal is not None and self.last_eth_signal == 0 and current_eth_signal == 1:
                self.play_sound()
            self.last_eth_signal = current_eth_signal
        
        # Son güncelleme
        self.update_label.config(
            text=f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    def start_monitoring(self):
        """İzlemeyi başlat"""
        def monitor():
            while self.running:
                try:
                    self.root.after(0, self.update_ui)
                    time.sleep(300)
                except Exception as e:
                    logger.error(f"İzleme hatası: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        
        # İlk güncelleme
        self.update_ui()
    
    def on_close(self):
        """Kapatma"""
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CryptoMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
