"""
Mobil Crypto Monitor - Kivy Uygulaması
GMSTR, BTC, ETH Canlı Monitör
"""

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import threading
import requests
import json

# Kivy versiyonu
kivy.require('2.1.0')

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


class CryptoCard(BoxLayout):
    def __init__(self, symbol, timeframe, color, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.symbol = symbol
        self.timeframe = timeframe
        self.color = color
        
        # Kart arka planı
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        # Başlık
        self.title = Label(
            text=f"{symbol}\n{timeframe}",
            font_size='16sp',
            bold=True,
            color=self.color,
            halign='center'
        )
        self.add_widget(self.title)
        
        # Fiyat
        self.price_label = Label(
            text="Yükleniyor...",
            font_size='14sp',
            color=(1, 1, 1, 1),
            halign='center'
        )
        self.add_widget(self.price_label)
        
        # Sinyal
        self.signal_label = Label(
            text="--",
            font_size='12sp',
            color=(1, 1, 1, 1),
            halign='center'
        )
        self.add_widget(self.signal_label)
        
        # Hedef
        self.target_label = Label(
            text="--",
            font_size='12sp',
            color=(1, 1, 1, 1),
            halign='center'
        )
        self.add_widget(self.target_label)
    
    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos
    
    def update_data(self, price, signal, target):
        self.price_label.text = f"₺${price:.2f}" if "GMSTR" in self.symbol else f"${price:.2f}"
        
        if signal == 1:
            signal_text = "📈 YÜKSELİŞ"
            signal_color = (0, 1, 0, 1)  # Yeşil
        else:
            signal_text = "📉 DÜŞÜŞ"
            signal_color = (1, 0, 0, 1)  # Kırmızı
        
        self.signal_label.text = signal_text
        self.signal_label.color = signal_color
        
        target_text = f"₺${target:.2f}" if "GMSTR" in self.symbol else f"${target:.2f}"
        self.target_label.text = f"Hedef: {target_text}"


class MobileCryptoMonitorApp(App):
    def __init__(self):
        super().__init__()
        self.sound_enabled = True
        self.last_btc_signal = None
        self.last_eth_signal = None
        
    def build(self):
        # Ana layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Başlık
        title_label = Label(
            text="GMSTR | BTC | ETH\nCANLI MONİTÖR",
            font_size='20sp',
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_y=None,
            height=dp(80)
        )
        main_layout.add_widget(title_label)
        
        # Ses butonu
        self.sound_button = Button(
            text="🔊 SES AÇIK",
            font_size='14sp',
            size_hint_y=None,
            height=dp(50),
            background_color=(0.3, 0.7, 0.3, 1)
        )
        self.sound_button.bind(on_press=self.toggle_sound)
        main_layout.add_widget(self.sound_button)
        
        # Kartlar grid
        cards_grid = GridLayout(cols=2, spacing=10, size_hint_y=0.7)
        
        # GMSTR kartları
        self.gmstr_4h_card = CryptoCard("GMSTR", "4 SAAT", (0, 1, 0, 1))
        self.gmstr_1d_card = CryptoCard("GMSTR", "1 GÜN", (0, 1, 0, 1))
        
        # BTC kartları
        self.btc_4h_card = CryptoCard("BTC", "4 SAAT", (1, 0.6, 0, 1))
        self.btc_1d_card = CryptoCard("BTC", "1 GÜN", (1, 0.6, 0, 1))
        
        # ETH kartları
        self.eth_4h_card = CryptoCard("ETH", "4 SAAT", (0.4, 0.5, 0.9, 1))
        self.eth_1d_card = CryptoCard("ETH", "1 GÜN", (0.4, 0.5, 0.9, 1))
        
        cards_grid.add_widget(self.gmstr_4h_card)
        cards_grid.add_widget(self.gmstr_1d_card)
        cards_grid.add_widget(self.btc_4h_card)
        cards_grid.add_widget(self.btc_1d_card)
        cards_grid.add_widget(self.eth_4h_card)
        cards_grid.add_widget(self.eth_1d_card)
        
        main_layout.add_widget(cards_grid)
        
        # Son güncelleme
        self.update_label = Label(
            text="Son Güncelleme: --",
            font_size='12sp',
            color=(0.5, 0.5, 0.5, 1),
            size_hint_y=None,
            height=dp(30)
        )
        main_layout.add_widget(self.update_label)
        
        # Arka plan rengi
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        # Otomatik güncelleme
        Clock.schedule_interval(self.update_data, 300)  # 5 dakikada bir
        
        # İlk güncelleme
        Clock.schedule_once(lambda dt: self.update_data(0), 1)
        
        return main_layout
    
    def toggle_sound(self, instance):
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.sound_button.text = "🔊 SES AÇIK"
            self.sound_button.background_color = (0.3, 0.7, 0.3, 1)
        else:
            self.sound_button.text = "🔇 SES KAPALI"
            self.sound_button.background_color = (0.7, 0.3, 0.3, 1)
    
    def play_sound(self):
        """Ses çal (mobil cihazlarda titreşim)"""
        if self.sound_enabled:
            try:
                # Android için titreşim
                if hasattr(self, 'vibrator'):
                    self.vibrator.vibrate(0.5)
            except Exception as e:
                logger.error(f"Ses/titreşim hatası: {e}")
    
    def get_binance_data(self, symbol):
        """Binance'den veri çek"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if response.status_code != 200 or not data:
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Verileri sayısal yap
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"Binance veri hatası ({symbol}): {e}")
            return None
    
    def get_gmstr_data(self):
        """GMSTR için örnek veri (gerçek BIST API gerekli)"""
        # Şimdilik örnek veri döndür
        dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
        np.random.seed(42)
        
        base_price = 1000
        price_changes = np.random.normal(0, 0.02, 720)
        prices = base_price * (1 + np.cumsum(price_changes))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 720)
        })
        
        return df
    
    def calculate_prediction(self, df, symbol_type):
        """Tahmin hesapla"""
        try:
            if df is None or df.empty:
                return None
            
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < 20:
                return None
            
            # Teknik göstergeler
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
            
            if symbol_type in ['btc', 'eth']:
                df['stochastic'] = calculate_stochastic(df)
                df['williams_r'] = calculate_williams_r(df)
                if symbol_type == 'eth':
                    df['obv'] = calculate_obv(df)
                    df['cci'] = calculate_cci(df)
            
            # Basit tahmin (gerçek model olmadan)
            current_price = df['close'].iloc[-1]
            
            # RSI ve momentum'a göre basit tahmin
            rsi = df['rsi'].iloc[-1]
            momentum = df['momentum'].iloc[-1]
            
            if rsi < 30 and momentum > 0:
                signal_4h = 1  # Al
                signal_1d = 1
                pred_4h = current_price * 1.02
                pred_1d = current_price * 1.05
            elif rsi > 70 and momentum < 0:
                signal_4h = 0  # Sat
                signal_1d = 0
                pred_4h = current_price * 0.98
                pred_1d = current_price * 0.95
            else:
                signal_4h = 1 if momentum > 0 else 0
                signal_1d = signal_4h
                pred_4h = current_price * (1.01 if signal_4h == 1 else 0.99)
                pred_1d = current_price * (1.03 if signal_1d == 1 else 0.97)
            
            return {
                'price': current_price,
                'pred_4h': pred_4h,
                'pred_1d': pred_1d,
                'signal_4h': signal_4h,
                'signal_1d': signal_1d
            }
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return None
    
    def update_data(self, dt):
        """Verileri güncelle"""
        try:
            # GMSTR
            gmstr_df = self.get_gmstr_data()
            gmstr_data = self.calculate_prediction(gmstr_df, 'gmstr')
            
            if gmstr_data:
                self.gmstr_4h_card.update_data(
                    gmstr_data['price'],
                    gmstr_data['signal_4h'],
                    gmstr_data['pred_4h']
                )
                self.gmstr_1d_card.update_data(
                    gmstr_data['price'],
                    gmstr_data['signal_1d'],
                    gmstr_data['pred_1d']
                )
            
            # BTC
            btc_df = self.get_binance_data('BTCUSDT')
            btc_data = self.calculate_prediction(btc_df, 'btc')
            
            if btc_data:
                self.btc_4h_card.update_data(
                    btc_data['price'],
                    btc_data['signal_4h'],
                    btc_data['pred_4h']
                )
                self.btc_1d_card.update_data(
                    btc_data['price'],
                    btc_data['signal_1d'],
                    btc_data['pred_1d']
                )
                
                # Sinyal değişimi kontrolü
                if self.last_btc_signal is not None and self.last_btc_signal == 0 and btc_data['signal_4h'] == 1:
                    self.play_sound()
                self.last_btc_signal = btc_data['signal_4h']
            
            # ETH
            eth_df = self.get_binance_data('ETHUSDT')
            eth_data = self.calculate_prediction(eth_df, 'eth')
            
            if eth_data:
                self.eth_4h_card.update_data(
                    eth_data['price'],
                    eth_data['signal_4h'],
                    eth_data['pred_4h']
                )
                self.eth_1d_card.update_data(
                    eth_data['price'],
                    eth_data['signal_1d'],
                    eth_data['pred_1d']
                )
                
                # Sinyal değişimi kontrolü
                if self.last_eth_signal is not None and self.last_eth_signal == 0 and eth_data['signal_4h'] == 1:
                    self.play_sound()
                self.last_eth_signal = eth_data['signal_4h']
            
            # Güncelleme zamanı
            self.update_label.text = f"Son Güncelleme: {datetime.now().strftime('%H:%M:%S')}"
            
        except Exception as e:
            logger.error(f"Veri güncelleme hatası: {e}")


if __name__ == "__main__":
    MobileCryptoMonitorApp().run()
