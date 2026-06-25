import time
import logging
import sys

from config import (
    PAPER_TRADING,
    SYMBOL_LIST,
    TIMEFRAME,
    POLL_INTERVAL_SECONDS,
    ATR_PERIOD,
    USE_TREND_FILTER,
    ENABLE_SHORT,
    VOLUME_THRESHOLD,
    VWAP_WINDOW,
    TIME_EXIT_BARS,
    EMA_TREND_PERIOD,
    SUPER_TREND_PERIOD,
    SUPER_TREND_MULTIPLIER,
    TIMEFRAME_1H,
    TRAILING_ATR_MULTIPLIER,
    ADX_THRESHOLD,
)
from exchange_client import ExchangeClient
from strategy import MultiIndicatorStrategy
from portfolio import PaperPortfolio
from gmstr_prediction_system import GMSTRPredictionSystem
from gmstr_enhanced.news_analyzer import get_analyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def process_symbol(client, strategy, portfolio, symbol, timeframe,
                     gmstr_system=None, news_analyzer=None):
    df = client.fetch_ohlcv(symbol, timeframe)
    if df.empty or len(df) < 50:
        return

    # Dual timeframe: 1h veri cek
    df_1h = client.fetch_ohlcv(symbol, TIMEFRAME_1H, limit=500)
    current_ts = df['timestamp'].iloc[-1]
    if not df_1h.empty and 'timestamp' in df_1h.columns:
        df_1h_window = df_1h[df_1h['timestamp'] <= current_ts]
        df_1h_window = df_1h_window if not df_1h_window.empty else None
    else:
        df_1h_window = None

    signal = strategy.get_signal(df, df_1h=df_1h_window)
    current_price = df['close'].iloc[-1]
    atr = strategy.get_last_atr(df)

    # ML + Haber Filtresi (GMSTR sembolleri icin)
    if gmstr_system and news_analyzer and "GMSTR" in symbol.upper():
        try:
            pred = gmstr_system.make_prediction(timeframe="4h")
            news = news_analyzer.fetch_news(count=12)
            sentiment = news_analyzer.get_news_sentiment_score(news)

            if pred:
                ml_confidence = pred.get('confidence', 0)
                ml_direction = pred.get('direction', '')

                if ml_confidence < 0.55:
                    logger.info(f"[{symbol}] ML guven dusuk ({ml_confidence:.2f}), sinyal iptal")
                    signal = "HOLD"
                elif (ml_direction == "YUKSELIS" and signal in ("SELL", "SHORT")) or \
                     (ml_direction == "DUSUS" and signal == "BUY"):
                    logger.info(f"[{symbol}] ML yonu ({ml_direction}) teknik sinyalle celisiyor, HOLD")
                    signal = "HOLD"

            if sentiment and signal == "BUY" and sentiment.get('overall_score', 0) < -0.5:
                logger.info(f"[{symbol}] Haber skoru negatif ({sentiment['overall_score']}), AL iptal")
                signal = "HOLD"
            if sentiment and signal in ("SELL", "SHORT") and sentiment.get('overall_score', 0) > 0.5:
                logger.info(f"[{symbol}] Haber skoru pozitif ({sentiment['overall_score']}), SAT iptal")
                signal = "HOLD"
        except Exception as e:
            logger.warning(f"[{symbol}] ML/Haber filtresi hatasi: {e}")

    total_value = portfolio.get_total_value(current_price)
    pos_type = "LONG" if portfolio.balance_asset > 0 else ("SHORT" if portfolio.short_position else "NONE")
    logger.info(
        f"[{symbol}] Fiyat: {current_price:.2f} | Sinyal: {signal} | Poz: {pos_type} | "
        f"Deger: {total_value:.2f} USDT"
    )

    # Piramit kontrolu: %2 kârda + SuperTrend AL sinyali
    if portfolio.balance_asset > 0:
        st_direction = strategy.get_super_trend_direction(df)
        portfolio.check_scaling_in(current_price, atr, len(df), st_direction)

    # Long Trailing Stop ve Time-Exit kontrolu
    if portfolio.balance_asset > 0:
        portfolio.update_trailing_stop(current_price, atr)
        if portfolio.check_stop_loss(current_price):
            portfolio.sell(current_price)
            return
        if portfolio.check_time_exit(len(df)):
            portfolio.sell(current_price)
            return

    # Short Trailing Stop ve Time-Exit kontrolu
    if portfolio.short_position:
        portfolio.update_trailing_stop(current_price, atr)
        if portfolio.check_stop_loss(current_price):
            portfolio.sell(current_price, is_short_close=True)
            return
        if portfolio.check_time_exit(len(df)):
            portfolio.sell(current_price, is_short_close=True)
            return

    # Sinyal islemleri
    # Trend Rider Agresif: Piramit + Chandelier Exit 2.5*ATR
    if signal == "BUY" and not portfolio.has_position():
        portfolio.buy(current_price, atr, bar_index=len(df))
    elif signal == "SHORT" and ENABLE_SHORT and not portfolio.has_position():
        portfolio.open_short(current_price, atr, bar_index=len(df))
    elif signal == "SELL" and portfolio.balance_asset > 0:
        portfolio.sell(current_price)


def main():
    logger.info("Bot baslatiliyor...")
    logger.info(f"Mod: {'Paper Trading' if PAPER_TRADING else 'Live Trading'}")
    logger.info(f"Timeframe: {TIMEFRAME} | Semboller: {SYMBOL_LIST}")

    if not PAPER_TRADING:
        logger.error("Live trading bu versiyonda devre disi. .env dosyasinda PAPER_TRADING=True yapin.")
        return

    client = ExchangeClient()
    strategy = MultiIndicatorStrategy(
        atr_period=ATR_PERIOD,
        volume_threshold=VOLUME_THRESHOLD,
        use_trend_filter=USE_TREND_FILTER,
        enable_short=ENABLE_SHORT,
        vwap_window=VWAP_WINDOW,
        ema_trend_period=EMA_TREND_PERIOD,
        super_trend_period=SUPER_TREND_PERIOD,
        super_trend_multiplier=SUPER_TREND_MULTIPLIER,
        adx_threshold=ADX_THRESHOLD,
    )

    portfolios = {symbol: PaperPortfolio() for symbol in SYMBOL_LIST}

    gmstr_system = GMSTRPredictionSystem()
    news_analyzer = get_analyzer()

    try:
        while True:
            for symbol in SYMBOL_LIST:
                try:
                    process_symbol(client, strategy, portfolios[symbol], symbol, TIMEFRAME,
                                   gmstr_system=gmstr_system, news_analyzer=news_analyzer)
                except Exception as e:
                    logger.exception(f"[{symbol}] Islem hatasi: {e}")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Kullanici tarafindan durduruldu.")
    except Exception as e:
        logger.exception(f"Beklenmeyen hata: {e}")


if __name__ == "__main__":
    main()
