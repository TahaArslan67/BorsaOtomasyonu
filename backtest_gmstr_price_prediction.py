"""
GMSTR Fiyat Tahmin Backtest
1-10 Mayıs 2026 ve 11 Mayıs 2026 için
"""

import numpy as np
import pandas as pd
import logging
from ai_trading_model_v4 import AITradingModelV4
from exchange_client_bist import BISTExchangeClient

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


def backtest_price_prediction(date_range):
    """Fiyat tahmin ile backtest"""
    symbol = "GMSTR"
    
    client = BISTExchangeClient()
    
    logger.info(f"Backtest: {symbol} | {date_range}")
    
    # Veri çek (son 2 gün)
    df = client.fetch_ohlcv(symbol, timeframe="1h", limit=48)
    
    if df is None or df.empty:
        logger.error(f"Veri çekilemedi: {symbol}")
        return False
    
    logger.info(f"Veri çekildi: {len(df)} bar")
    
    # Inf değerlerini temizle
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    logger.info(f"Veri temizlendi: {len(df)} bar")
    
    # Features hesapla
    df['rsi'] = calculate_rsi(df)
    df['atr'] = calculate_atr(df)
    df['z_score'] = calculate_z_score(df)
    df['volume_delta'] = calculate_volume_delta(df)
    df['macd'] = calculate_macd(df)
    df['ema'] = calculate_ema(df)
    
    # Fiyat tahmin modellerini yükle
    import joblib
    model_1h = joblib.load(f"price_prediction_{symbol}_1h.pkl")
    model_4h = joblib.load(f"price_prediction_{symbol}_4h.pkl")
    model_1d = joblib.load(f"price_prediction_{symbol}_1d.pkl")
    
    # Features
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema']
    
    # Backtest parametreleri
    initial_balance = 500.0
    balance = initial_balance
    risk_per_trade = 0.005
    commission_rate = 0.001
    slippage_rate = 0.0005
    
    trades = []
    position = 0.0
    entry_price = 0.0
    
    logger.info("Fiyat tahminleri yapılıyor...")
    
    for i in range(len(df) - 1):
        current_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(atr) or atr == 0:
            continue
        
        # Fiyat tahminleri yap
        if i >= 20:
            X = df[features].iloc[i:i+1]
            
            try:
                pred_1h = model_1h.predict(X)[0]
                pred_4h = model_4h.predict(X)[0]
                pred_1d = model_1d.predict(X)[0]
                
                # Tahminler
                logger.info(f"Zaman: {df['timestamp'].iloc[i]} | Fiyat: ${current_price:.2f}")
                logger.info(f"  1h Tahmin: ${pred_1h:.2f} | Değişim: %{((pred_1h - current_price) / current_price) * 100:.2f}")
                logger.info(f"  4h Tahmin: ${pred_4h:.2f} | Değişim: %{((pred_4h - current_price) / current_price) * 100:.2f}")
                logger.info(f"  1gün Tahmin: ${pred_1d:.2f} | Değişim: %{((pred_1d - current_price) / current_price) * 100:.2f}")
                
                # 1h tahmine göre işlem yap
                if pred_1h > current_price * 1.005:  # %0.5 yükseliş bekleniyor
                    if position == 0:
                        risk_capital = balance * risk_per_trade
                        stop_distance = atr * 2.0  # Daha geniş stop loss
                        position_size = risk_capital / stop_distance
                        
                        buy_price = current_price * (1 + slippage_rate)
                        cost = position_size * buy_price
                        commission = cost * commission_rate
                        total_cost = cost + commission
                        
                        if total_cost <= balance:
                            balance -= total_cost
                            position = position_size
                            entry_price = buy_price
                            
                            trades.append({
                                'type': 'BUY',
                                'timestamp': df['timestamp'].iloc[i],
                                'price': buy_price,
                                'amount': position,
                                'balance': balance,
                            })
                            
                            logger.info(f"BUY | Fiyat: ${buy_price:.2f} | Miktar: {position:.6f} | Kasa: ${balance:.2f}")
                
                # Pozisyon kapatma kontrolü
                if position > 0:
                    if pred_1h < current_price * 0.995:  # %0.5 düşüş bekleniyor
                        sell_price = current_price * (1 - slippage_rate)
                        revenue = position * sell_price
                        commission = revenue * commission_rate
                        net_revenue = revenue - commission
                        profit = net_revenue - (position * entry_price)
                        balance += net_revenue
                        
                        trades.append({
                            'type': 'SELL',
                            'timestamp': df['timestamp'].iloc[i],
                            'price': sell_price,
                            'amount': position,
                            'profit': profit,
                            'balance': balance,
                        })
                        
                        logger.info(f"SELL | Fiyat: ${sell_price:.2f} | Miktar: {position:.6f} | Kar/Zarar: ${profit:.2f} | Kasa: ${balance:.2f}")
                        
                        position = 0.0
                        entry_price = 0.0
            except Exception as e:
                continue
    
    # Sonuçları hesapla
    total_profit = balance - initial_balance
    profit_percentage = (total_profit / initial_balance) * 100
    
    logger.info("=" * 60)
    logger.info(f"BACKTEST SONUÇLARI - {date_range}")
    logger.info("=" * 60)
    logger.info(f"Başlangıç Bakiyesi: ${initial_balance:.2f}")
    logger.info(f"Son Bakiye: ${balance:.2f}")
    logger.info(f"Toplam Kar/Zarar: ${total_profit:.2f}")
    logger.info(f"Kar/Zarar Yüzdesi: %{profit_percentage:.2f}")
    logger.info(f"Toplam İşlem: {len(trades)}")
    
    # Kazançlı ve zararlı işlemler
    winning_trades = [t for t in trades if 'profit' in t and t['profit'] > 0]
    losing_trades = [t for t in trades if 'profit' in t and t['profit'] < 0]
    
    logger.info(f"Kazançlı İşlemler: {len(winning_trades)}")
    logger.info(f"Zararlı İşlemler: {len(losing_trades)}")
    
    if len(winning_trades) > 0:
        avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades)
        logger.info(f"Ortalama Kazanç: ${avg_win:.2f}")
    
    if len(losing_trades) > 0:
        avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades)
        logger.info(f"Ortalama Zarar: ${avg_loss:.2f}")
    
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    # 1-10 Mayıs 2026
    backtest_price_prediction("1-10 Mayıs 2026")
    
    # 11 Mayıs 2026
    backtest_price_prediction("11 Mayıs 2026")
