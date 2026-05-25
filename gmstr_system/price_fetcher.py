"""
GMSTR Canlı Fiyat Çekici - Yahoo Finance API
Her 5 dakikada son fiyatı ve son saatlik verileri çeker.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict


class GMSTRPriceFetcher:
    """Yahoo Finance üzerinden GMSTR.IS canlı veri çekici."""

    TICKER = "GMSTR.IS"

    @classmethod
    def fetch_live_price(cls) -> Optional[float]:
        """Son canlı fiyatı döndür."""
        try:
            ticker = yf.Ticker(cls.TICKER)
            # Önce info'dan deneyelim
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price:
                return float(price)

            # Olmazsa son history'den al
            hist = ticker.history(period="1d", interval="1m")
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
            return None
        except Exception as e:
            print(f"[PriceFetcher] Canlı fiyat hatası: {e}")
            return None

    @classmethod
    def fetch_recent_history(cls, period: str = "5d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Son N günlük saatlik veriyi döndür."""
        try:
            ticker = yf.Ticker(cls.TICKER)
            hist = ticker.history(period=period, interval=interval)
            if len(hist) == 0:
                return None
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]
            # datetime -> date olarak kullan
            if 'datetime' in hist.columns:
                hist = hist.rename(columns={'datetime': 'date'})
            return hist
        except Exception as e:
            print(f"[PriceFetcher] History hatası: {e}")
            return None

    @classmethod
    def fetch_daily_history(cls, period: str = "2y") -> Optional[pd.DataFrame]:
        """Günlük veri çek (model eğitimine benzer formatta)."""
        try:
            ticker = yf.Ticker(cls.TICKER)
            hist = ticker.history(period=period, interval="1d")
            if len(hist) == 0:
                return None
            hist = hist.reset_index()
            hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]
            if 'datetime' in hist.columns:
                hist = hist.rename(columns={'datetime': 'date'})
            return hist
        except Exception as e:
            print(f"[PriceFetcher] Daily history hatası: {e}")
            return None

    @classmethod
    def get_price_with_fallback(cls) -> Dict:
        """Canlı fiyat çek, başarısız olursa CSV'den fallback."""
        price = cls.fetch_live_price()
        source = "API (Yahoo Finance)"
        if price is None:
            # Fallback: gercek_data.csv'den son değer
            try:
                from pathlib import Path
                import sys
                base = Path.cwd()
                if hasattr(sys, '_MEIPASS'):
                    base = Path(sys._MEIPASS)
                csv_path = base / 'claude' / 'gercek_data.csv'
                if csv_path.exists():
                    import pandas as pd
                    raw = pd.read_csv(csv_path, header=None)
                    raw.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    price = float(raw['Close'].iloc[-1])
                    source = "CSV (yerel)"
            except Exception:
                pass
        return {
            'price': price,
            'source': source,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
