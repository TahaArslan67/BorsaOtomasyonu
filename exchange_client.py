import ccxt
import pandas as pd
import logging
import time
import socket
import urllib.request
from functools import wraps
from config import BINANCE_API_KEY, BINANCE_SECRET_KEY, PAPER_TRADING

logger = logging.getLogger(__name__)


def is_internet_available(host="https://api.binance.com", timeout=3):
    try:
        urllib.request.urlopen(host, timeout=timeout)
        return True
    except Exception:
        return False


def retry_on_network_error(max_retries=3, backoff_seconds=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    if not is_internet_available():
                        logger.warning("Internet baglantisi yok, bekleniyor...")
                        time.sleep(backoff_seconds)
                        continue
                    return func(*args, **kwargs)
                except ccxt.NetworkError as e:
                    logger.warning(f"Ag hatasi (deneme {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(backoff_seconds * attempt)
                    else:
                        logger.error("Maksimum deneme sayisina ulasildi.")
                        return None
                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"Rate limit asildi (deneme {attempt}/{max_retries}): {e}")
                    if attempt < max_retries:
                        time.sleep(backoff_seconds * attempt * 2)
                    else:
                        logger.error("Maksimum deneme sayisina ulasildi.")
                        return None
        return wrapper
    return decorator


class ExchangeClient:
    def __init__(self, sandbox_mode=None, use_credentials=True):
        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        }
        if use_credentials:
            config['apiKey'] = BINANCE_API_KEY
            config['secret'] = BINANCE_SECRET_KEY
        self.exchange = ccxt.binance(config)

        if sandbox_mode is None:
            sandbox_mode = PAPER_TRADING
        if sandbox_mode and use_credentials:
            self.exchange.set_sandbox_mode(True)
            logger.info("Paper Trading modu aktif (sandbox).")

    @retry_on_network_error(max_retries=3, backoff_seconds=5)
    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except ccxt.ExchangeError as e:
            logger.error(f"Borsa hatasi (OHLCV): {e}")
            return pd.DataFrame()

    def fetch_balance(self):
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Bakiye cekilemedi: {e}")
            return {}

    def create_market_buy_order(self, symbol, amount):
        try:
            order = self.exchange.create_market_buy_order(symbol, amount)
            logger.info(f"Gercek AL emri gonderildi: {order}")
            return order
        except Exception as e:
            logger.error(f"AL emri hatasi: {e}")
            return None

    def create_market_sell_order(self, symbol, amount):
        try:
            order = self.exchange.create_market_sell_order(symbol, amount)
            logger.info(f"Gercek SAT emri gonderildi: {order}")
            return order
        except Exception as e:
            logger.error(f"SAT emri hatasi: {e}")
            return None
