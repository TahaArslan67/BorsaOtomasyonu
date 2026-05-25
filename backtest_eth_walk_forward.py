"""
ETH Walk-Forward Test
"""

import numpy as np
import pandas as pd
import logging
from exchange_client_crypto import CryptoExchangeClient

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


def backtest_eth_walk_forward():
    """ETH walk-forward test"""
    symbol = "ETHUSDT"
    
    client = CryptoExchangeClient()
    
    logger.info(f"ETH walk-forward test: {symbol}")
    
    # Veri çek (son 1000 bar)
    df = client.fetch_ohlcv(symbol, timeframe="1h", limit=1000)
    
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
    df['stochastic'] = calculate_stochastic(df)
    df['williams_r'] = calculate_williams_r(df)
    df['obv'] = calculate_obv(df)
    df['cci'] = calculate_cci(df)
    
    # Walk-forward test
    train_size = 500
    test_size = 500
    
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:train_size+test_size]
    
    logger.info(f"Eğitim verisi: {len(train_data)} bar")
    logger.info(f"Test verisi: {len(test_data)} bar")
    
    # Features
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                'stochastic', 'williams_r', 'obv', 'cci']
    
    # Backtest parametreleri
    initial_balance = 10000
    balance = initial_balance
    position_size = 0.1
    commission = 0.001
    slippage = 0.0005
    
    trades_4h = []
    trades_1d = []
    
    logger.info("=" * 60)
    logger.info("ETH WALK-FORWARD TEST")
    logger.info("=" * 60)
    
    # Modelleri yükle
    import joblib
    model_4h = joblib.load(f"price_prediction_ETH_4h.pkl")
    model_1d = joblib.load(f"price_prediction_ETH_1d.pkl")
    
    # 4h backtest
    for i in range(len(test_data) - 4):
        X = test_data[features].iloc[i:i+1]
        current_price = test_data['close'].iloc[i]
        
        if i + 4 < len(test_data):
            pred_price_4h = model_4h.predict(X)[0]
            actual_price_4h = test_data['close'].iloc[i+4]
            
            # Sinyal
            signal = 1 if pred_price_4h > current_price else -1
            
            # Kar/Zarar hesapla
            if signal == 1:
                profit = (actual_price_4h - current_price) * position_size
            else:
                profit = (current_price - actual_price_4h) * position_size
            
            # Komisyon ve slippage
            profit -= (current_price * position_size * commission)
            profit -= (current_price * position_size * slippage)
            
            balance += profit
            
            trades_4h.append({
                'entry': current_price,
                'exit': actual_price_4h,
                'signal': 'BUY' if signal == 1 else 'SELL',
                'profit': profit,
                'balance': balance
            })
    
    # 1gün backtest
    balance_1d = initial_balance
    
    for i in range(len(test_data) - 24):
        X = test_data[features].iloc[i:i+1]
        current_price = test_data['close'].iloc[i]
        
        if i + 24 < len(test_data):
            pred_price_1d = model_1d.predict(X)[0]
            actual_price_1d = test_data['close'].iloc[i+24]
            
            # Sinyal
            signal = 1 if pred_price_1d > current_price else -1
            
            # Kar/Zarar hesapla
            if signal == 1:
                profit = (actual_price_1d - current_price) * position_size
            else:
                profit = (current_price - actual_price_1d) * position_size
            
            # Komisyon ve slippage
            profit -= (current_price * position_size * commission)
            profit -= (current_price * position_size * slippage)
            
            balance_1d += profit
            
            trades_1d.append({
                'entry': current_price,
                'exit': actual_price_1d,
                'signal': 'BUY' if signal == 1 else 'SELL',
                'profit': profit,
                'balance': balance_1d
            })
    
    # Sonuçlar
    profit_4h = balance - initial_balance
    profit_pct_4h = (profit_4h / initial_balance) * 100
    
    profit_1d = balance_1d - initial_balance
    profit_pct_1d = (profit_1d / initial_balance) * 100
    
    win_trades_4h = len([t for t in trades_4h if t['profit'] > 0])
    loss_trades_4h = len([t for t in trades_4h if t['profit'] <= 0])
    win_rate_4h = (win_trades_4h / len(trades_4h)) * 100 if trades_4h else 0
    
    win_trades_1d = len([t for t in trades_1d if t['profit'] > 0])
    loss_trades_1d = len([t for t in trades_1d if t['profit'] <= 0])
    win_rate_1d = (win_trades_1d / len(trades_1d)) * 100 if trades_1d else 0
    
    logger.info("=" * 60)
    logger.info("4h WALK-FORWARD BACKTEST SONUÇLARI")
    logger.info("=" * 60)
    logger.info(f"Toplam İşlem: {len(trades_4h)}")
    logger.info(f"Kazançlı İşlem: {win_trades_4h}")
    logger.info(f"Kayıplı İşlem: {loss_trades_4h}")
    logger.info(f"Kazanma Oranı: %{win_rate_4h:.2f}")
    logger.info(f"Net Kar: ${profit_4h:.2f}")
    logger.info(f"Kar Oranı: %{profit_pct_4h:.2f}")
    logger.info(f"Son Bakiye: ${balance:.2f}")
    logger.info("=" * 60)
    
    logger.info("=" * 60)
    logger.info("1gün WALK-FORWARD BACKTEST SONUÇLARI")
    logger.info("=" * 60)
    logger.info(f"Toplam İşlem: {len(trades_1d)}")
    logger.info(f"Kazançlı İşlem: {win_trades_1d}")
    logger.info(f"Kayıplı İşlem: {loss_trades_1d}")
    logger.info(f"Kazanma Oranı: %{win_rate_1d:.2f}")
    logger.info(f"Net Kar: ${profit_1d:.2f}")
    logger.info(f"Kar Oranı: %{profit_pct_1d:.2f}")
    logger.info(f"Son Bakiye: ${balance_1d:.2f}")
    logger.info("=" * 60)
    
    if profit_pct_4h > 0:
        logger.info("4h modeli kârlı ✓")
    else:
        logger.info("4h modeli zararlı ✗")
    
    if profit_pct_1d > 0:
        logger.info("1gün modeli kârlı ✓")
    else:
        logger.info("1gün modeli zararlı ✗")
    
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    backtest_eth_walk_forward()
