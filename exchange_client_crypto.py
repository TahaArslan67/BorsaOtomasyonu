"""
Kripto Exchange Client
Bitcoin (BTC) için
"""

import requests
import pandas as pd
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class CryptoExchangeClient:
    """Kripto exchange client"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = "BTCUSDT"
    
    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        """OHLCV verisi çek"""
        try:
            # Binance API timeframe mapping
            timeframe_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
                "1w": "1w",
            }
            
            binance_timeframe = timeframe_map.get(timeframe, "1h")
            
            params = {
                "symbol": symbol,
                "interval": binance_timeframe,
                "limit": limit
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Gereksiz sütunları sil
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Veri çekildi: {symbol} | {len(df)} bar")
            
            return df
            
        except Exception as e:
            logger.error(f"Veri çekme hatası: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Şu anki fiyatı al"""
        try:
            params = {"symbol": symbol}
            response = requests.get(f"{self.base_url}/ticker/price", params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            price = float(data['price'])
            
            logger.info(f"Fiyat alındı: {symbol} | ${price:.2f}")
            
            return price
            
        except Exception as e:
            logger.error(f"Fiyat alma hatası: {e}")
            return None


if __name__ == "__main__":
    client = CryptoExchangeClient()
    
    # Test
    df = client.fetch_ohlcv("BTCUSDT", "1h", 100)
    if df is not None:
        print(df.head())
    
    price = client.get_current_price("BTCUSDT")
    if price is not None:
        print(f"Şu anki fiyat: ${price:.2f}")
