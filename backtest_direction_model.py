"""
GMSTR Yön Tahmin Modeli Backtest
4h ve 1gün için
"""

import numpy as np
import pandas as pd
import logging
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


def backtest_direction_model():
    """GMSTR yön tahmin modeli backtest"""
    symbol = "GMSTR"
    
    client = BISTExchangeClient()
    
    logger.info(f"Backtest: {symbol} (4h ve 1gün)")
    
    # Veri çek (son 1 ay)
    df = client.fetch_ohlcv(symbol, timeframe="1h", limit=720)
    
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
    df['bollinger_upper'] = calculate_bollinger_upper(df)
    df['bollinger_lower'] = calculate_bollinger_lower(df)
    df['sma'] = calculate_sma(df)
    df['momentum'] = calculate_momentum(df)
    
    # Yön tahmin modellerini yükle
    import joblib
    model_4h = joblib.load(f"direction_prediction_{symbol}_4h.pkl")
    model_1d = joblib.load(f"direction_prediction_{symbol}_1d.pkl")
    
    # Features
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
    
    # Backtest parametreleri
    initial_balance = 500.0
    balance = initial_balance
    risk_per_trade = 0.005
    commission_rate = 0.001
    slippage_rate = 0.0005
    
    trades = []
    position = 0.0
    entry_price = 0.0
    position_type = None  # '4h' veya '1d'
    
    logger.info("=" * 60)
    logger.info("BACKTEST BAŞLIYOR")
    logger.info("=" * 60)
    
    for i in range(len(df) - 1):
        current_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(atr) or atr == 0:
            continue
        
        # Stop loss kontrolü
        if position > 0:
            stop_loss_price = entry_price - (atr * 2.0)
            
            if df['low'].iloc[i] <= stop_loss_price:
                sell_price = current_price * (1 - slippage_rate)
                revenue = position * sell_price
                commission = revenue * commission_rate
                net_revenue = revenue - commission
                profit = net_revenue - (position * entry_price)
                balance += net_revenue
                
                trades.append({
                    'type': 'SELL (STOP)',
                    'timestamp': df['timestamp'].iloc[i],
                    'price': sell_price,
                    'amount': position,
                    'profit': profit,
                    'balance': balance,
                    'position_type': position_type,
                })
                
                logger.info(f"SELL (STOP) | {position_type} | Fiyat: ${sell_price:.2f} | Kar/Zarar: ${profit:.2f} | Kasa: ${balance:.2f}")
                
                position = 0.0
                entry_price = 0.0
                position_type = None
                continue
        
        # Take profit kontrolü (4h için)
        if position > 0 and position_type == '4h':
            take_profit_price = entry_price + (atr * 3.0)
            
            if df['high'].iloc[i] >= take_profit_price:
                sell_price = current_price * (1 - slippage_rate)
                revenue = position * sell_price
                commission = revenue * commission_rate
                net_revenue = revenue - commission
                profit = net_revenue - (position * entry_price)
                balance += net_revenue
                
                trades.append({
                    'type': 'SELL (TP)',
                    'timestamp': df['timestamp'].iloc[i],
                    'price': sell_price,
                    'amount': position,
                    'profit': profit,
                    'balance': balance,
                    'position_type': position_type,
                })
                
                logger.info(f"SELL (TP) | {position_type} | Fiyat: ${sell_price:.2f} | Kar/Zarar: ${profit:.2f} | Kasa: ${balance:.2f}")
                
                position = 0.0
                entry_price = 0.0
                position_type = None
                continue
        
        # Take profit kontrolü (1gün için)
        if position > 0 and position_type == '1d':
            take_profit_price = entry_price + (atr * 5.0)
            
            if df['high'].iloc[i] >= take_profit_price:
                sell_price = current_price * (1 - slippage_rate)
                revenue = position * sell_price
                commission = revenue * commission_rate
                net_revenue = revenue - commission
                profit = net_revenue - (position * entry_price)
                balance += net_revenue
                
                trades.append({
                    'type': 'SELL (TP)',
                    'timestamp': df['timestamp'].iloc[i],
                    'price': sell_price,
                    'amount': position,
                    'profit': profit,
                    'balance': balance,
                    'position_type': position_type,
                })
                
                logger.info(f"SELL (TP) | {position_type} | Fiyat: ${sell_price:.2f} | Kar/Zarar: ${profit:.2f} | Kasa: ${balance:.2f}")
                
                position = 0.0
                entry_price = 0.0
                position_type = None
                continue
        
        # Sinyal kontrolü
        if position == 0 and i >= 20:
            X = df[features].iloc[i:i+1]
            
            # 4h sinyali
            signal_4h = model_4h.predict(X)[0]
            
            if signal_4h == 1:  # YÜKSELİŞ bekleniyor
                risk_capital = balance * risk_per_trade
                stop_distance = atr * 2.0
                position_size = risk_capital / stop_distance
                
                buy_price = current_price * (1 + slippage_rate)
                cost = position_size * buy_price
                commission = cost * commission_rate
                total_cost = cost + commission
                
                if total_cost <= balance:
                    balance -= total_cost
                    position = position_size
                    entry_price = buy_price
                    position_type = '4h'
                    
                    trades.append({
                        'type': 'BUY',
                        'timestamp': df['timestamp'].iloc[i],
                        'price': buy_price,
                        'amount': position,
                        'balance': balance,
                        'position_type': position_type,
                    })
                    
                    logger.info(f"BUY | 4h | Fiyat: ${buy_price:.2f} | Miktar: {position:.6f} | Kasa: ${balance:.2f}")
            
            # 1gün sinyali
            signal_1d = model_1d.predict(X)[0]
            
            if signal_1d == 1:  # YÜKSELİŞ bekleniyor
                risk_capital = balance * risk_per_trade
                stop_distance = atr * 2.0
                position_size = risk_capital / stop_distance
                
                buy_price = current_price * (1 + slippage_rate)
                cost = position_size * buy_price
                commission = cost * commission_rate
                total_cost = cost + commission
                
                if total_cost <= balance:
                    balance -= total_cost
                    position = position_size
                    entry_price = buy_price
                    position_type = '1d'
                    
                    trades.append({
                        'type': 'BUY',
                        'timestamp': df['timestamp'].iloc[i],
                        'price': buy_price,
                        'amount': position,
                        'balance': balance,
                        'position_type': position_type,
                    })
                    
                    logger.info(f"BUY | 1gün | Fiyat: ${buy_price:.2f} | Miktar: {position:.6f} | Kasa: ${balance:.2f}")
    
    # Sonuçları hesapla
    total_profit = balance - initial_balance
    profit_percentage = (total_profit / initial_balance) * 100
    
    logger.info("=" * 60)
    logger.info("BACKTEST SONUÇLARI")
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
    backtest_direction_model()
