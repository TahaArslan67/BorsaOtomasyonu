import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() == "true"
SYMBOL_LIST = [s.strip() for s in os.getenv("SYMBOL_LIST", "PAXG/USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "1h")

INITIAL_BALANCE_USDT = float(os.getenv("INITIAL_BALANCE_USDT", "10000.0"))
LEVERAGE = float(os.getenv("LEVERAGE", "125.0"))
POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", "100.0"))

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "40.0"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "60.0"))

STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "2.0"))

COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))
TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "True").lower() == "true"
TRAILING_STOP_ACTIVATION_PERCENT = float(os.getenv("TRAILING_STOP_ACTIVATION_PERCENT", "1.0"))
MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "2.0"))

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
TRAILING_ATR_MULTIPLIER = float(os.getenv("TRAILING_ATR_MULTIPLIER", "6.0"))
ATR_FILTER_MULTIPLIER = float(os.getenv("ATR_FILTER_MULTIPLIER", "0.8"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_THRESHOLD = float(os.getenv("ADX_THRESHOLD", "25.0"))  # Güçlü trend için ADX eşiği
# ADX kaldırıldı - RSI pullback modeli
USE_TREND_FILTER = os.getenv("USE_TREND_FILTER", "False").lower() == "true"
ENABLE_SHORT = os.getenv("ENABLE_SHORT", "True").lower() == "true"
VOLUME_THRESHOLD = float(os.getenv("VOLUME_THRESHOLD", "0.5"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "3.0"))

# VWAP + EMA Trend + SuperTrend
VWAP_WINDOW = int(os.getenv("VWAP_WINDOW", "100"))
EMA_TREND_PERIOD = int(os.getenv("EMA_TREND_PERIOD", "200"))
SUPER_TREND_PERIOD = int(os.getenv("SUPER_TREND_PERIOD", "20"))
SUPER_TREND_MULTIPLIER = float(os.getenv("SUPER_TREND_MULTIPLIER", "4.0"))
TIMEFRAME_1H = os.getenv("TIMEFRAME_1H", "1h")

# Stop sonrasi bekleme (cool-down)
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "0"))

# Time-Exit: Trend Rider Agresif - 72 bar = 6 saat (büyük hareketlere zaman tanı)
TIME_EXIT_BARS = int(os.getenv("TIME_EXIT_BARS", "9999"))

POLL_INTERVAL_SECONDS = 15

# Trend Rider: Sabit TP yok, trailing stop aktif
FIXED_TAKE_PROFIT = os.getenv("FIXED_TAKE_PROFIT", "True").lower() == "true"
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "10.0"))
