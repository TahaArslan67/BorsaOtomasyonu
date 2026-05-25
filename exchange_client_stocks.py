"""
Stock & Commodity Exchange Client
Yahoo Finance API kullanarak hisse senedi ve emtia verisi çeker
"""

import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StockCommodityExchangeClient:
    """Stock & Commodity Exchange Client"""
    
    def __init__(self):
        self.sandbox_mode = False
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 2190):
        """
        OHLCV verisi çeker
        
        Args:
            symbol: Hisse senedi veya emtia sembolü (örn: AAPL, MSFT, GOLD)
            timeframe: Timeframe (örn: 1h, 4h, 1d)
            limit: Bar sayısı
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Yahoo Finance symbol format
            ticker = yf.Ticker(symbol)
            
            # Timeframe mapping
            timeframe_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
                "1w": "1wk",
                "1M": "1mo"
            }
            
            interval = timeframe_map.get(timeframe, "1h")
            
            # Veri çek
            if timeframe == "15m":
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=59)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period="1y", interval=interval)
            
            if df.empty:
                logger.error(f"Veri bulunamadı: {symbol}")
                return None
            
            # DataFrame'i formatla
            df = df.reset_index()
            
            # Tarih sütunu adını kontrol et
            date_column = None
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_column = col
                    break
            
            if date_column is None:
                logger.error(f"Tarih sütunu bulunamadı: {symbol}")
                return None
            
            df['timestamp'] = pd.to_datetime(df[date_column]).astype('int64') // 10**9
            
            # Sütun adlarını kontrol et ve formatla
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    column_mapping[col] = 'open'
                elif 'high' in col_lower:
                    column_mapping[col] = 'high'
                elif 'low' in col_lower:
                    column_mapping[col] = 'low'
                elif 'close' in col_lower:
                    column_mapping[col] = 'close'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'volume'
            
            df = df.rename(columns=column_mapping)
            
            # Sütunları seç
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Eksik sütunlar: {symbol} | {missing_columns}")
                return None
            
            df = df[required_columns]
            
            # Limit uygula
            df = df.tail(limit)
            
            logger.info(f"Veri çekildi: {symbol} | {len(df)} bar")
            return df
            
        except Exception as e:
            logger.error(f"Veri çekme hatası: {symbol} | {e}")
            return None
