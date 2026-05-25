"""
GMSTR Tüm Zaman Dilimleri Backtest
1h, 4h, 1d, 1w için yön doğruluğu, fiyat doğruluğu, yüzde kazancı hesapla
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import logging
import yfinance as yf
import joblib
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Teknik göstergeler
def calculate_rsi(df, w=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(w).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 50)

def calculate_atr(df, w=14):
    tr = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    return tr.rolling(w).mean()

def calculate_z_score(df, w=20):
    ma = df['close'].rolling(w).mean()
    std = df['close'].rolling(w).std()
    return ((df['close'] - ma) / std.replace(0, np.nan)).replace([np.inf, -np.inf], 0)

def calculate_volume_delta(df, w=20):
    return df['volume'].pct_change(w).replace([np.inf, -np.inf], 0)

def calculate_macd(df, fast=12, slow=26):
    ema_f = df['close'].ewm(span=fast).mean()
    ema_s = df['close'].ewm(span=slow).mean()
    return (ema_f - ema_s).replace([np.inf, -np.inf], 0)

def calculate_ema(df, w=20):
    return df['close'].ewm(span=w).mean()

def calculate_bollinger_upper(df, w=20):
    ma = df['close'].rolling(w).mean()
    std = df['close'].rolling(w).std()
    return (ma + 2*std).replace([np.inf, -np.inf], np.nan).fillna(df['close'])

def calculate_bollinger_lower(df, w=20):
    ma = df['close'].rolling(w).mean()
    std = df['close'].rolling(w).std()
    return (ma - 2*std).replace([np.inf, -np.inf], np.nan).fillna(df['close'])

def calculate_sma(df, w=20):
    return df['close'].rolling(w).mean()

def calculate_momentum(df, w=10):
    return df['close'] - df['close'].shift(w)

def compute_all(df):
    df = df.copy()
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
    return df.ffill().bfill()

def backtest_tf(tf, model_path, period='1y', bars_per_period=24):
    """
    tf: '1h', '4h', '1d', '1w'
    period: test süresi
    bars_per_period: her tf için bir sonraki periyotta kaç bar sonra fiyat
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST: {tf}")
    logger.info(f"{'='*60}")
    
    # Model yükle
    model = joblib.load(model_path)
    features = list(model.feature_names_in_)
    
    # Veri çek
    ticker = yf.Ticker("GMSTR.IS")
    df_daily = ticker.history(period=period, interval="1d")
    
    # Zaman dilimine göre resample
    if tf == '1h':
        # Günlük veriyi saatlik olarak tut
        df = df_daily.copy()
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        bars_per_period = 1  # 1 saat sonra
    elif tf == '4h':
        # Günlük veriyi 4 saatlik olarak resample
        df_daily['Date'] = df_daily.index
        df = df_daily.resample('4h', on='Date').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna().reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        bars_per_period = 4  # 4 saat sonra
    elif tf == '1d':
        df = df_daily.copy()
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        bars_per_period = 1  # 1 gün sonra
    elif tf == '1w':
        df_daily['Date'] = df_daily.index
        df = df_daily.resample('W-FRI', on='Date').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna().reset_index()
        df['timestamp'] = df['Date'].astype('int64') // 10**9
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        bars_per_period = 1  # 1 hafta sonra
    
    if df.empty or len(df) < 50:
        logger.error(f"Yetersiz veri: {len(df)} bar")
        return None
    
    # Teknik göstergeler
    df = compute_all(df)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Hedef: N bar sonra fiyat
    df['target'] = df['close'].shift(-bars_per_period)
    df = df.dropna(subset=['target'])
    
    X = df[features]
    y = df['target']
    
    # Tahminler
    y_pred = model.predict(X)
    
    # Yön doğruluğu: tahmin edilen yön vs gerçek yön
    # Yön: bugünkü fiyata göre hedef fiyatın yönü
    current_prices = df['close'].values
    actual_direction = (y.values > current_prices)  # Gerçek yön
    pred_direction = (y_pred > current_prices)  # Tahmin edilen yön
    
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Fiyat doğruluğu (MAE ve MAPE)
    mae = np.mean(np.abs(y_pred - y.values))
    mape = np.mean(np.abs((y_pred - y.values) / y.values)) * 100
    
    # Yüzde kazanç: Her sinyalde al-sat yaparsak
    # AL sinyali: pred > current -> al, N bar sonra sat
    # SAT sinyali: pred <= current -> short, N bar sonra kapat
    initial_capital = 10000.0
    capital = initial_capital
    trades = []
    
    for i in range(len(X)):
        if pred_direction[i]:  # AL sinyali
            entry_price = current_prices[i]
            exit_price = y.values[i]
            pct_change = (exit_price - entry_price) / entry_price
            capital *= (1 + pct_change)
            trades.append({
                'type': 'BUY',
                'entry': entry_price,
                'exit': exit_price,
                'pct': pct_change * 100
            })
        else:  # SAT sinyali (short veya beklemek)
            # Short yapmıyoruz, sadece bekliyoruz
            # Eğer short yapılsaydı:
            entry_price = current_prices[i]
            exit_price = y.values[i]
            pct_change = (entry_price - exit_price) / entry_price  # Short karı
            capital *= (1 + pct_change)
            trades.append({
                'type': 'SELL',
                'entry': entry_price,
                'exit': exit_price,
                'pct': pct_change * 100
            })
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    # Sonuçları özetle
    logger.info(f"Veri: {len(df)} bar, Test süresi: {period}")
    logger.info(f"Toplam işlem: {len(trades)}")
    logger.info(f"AL sinyali: {sum(1 for t in trades if t['type'] == 'BUY')}")
    logger.info(f"SAT sinyali: {sum(1 for t in trades if t['type'] == 'SELL')}")
    logger.info(f"Yön Doğruluğu: %{direction_accuracy:.2f}")
    logger.info(f"Fiyat Doğruluğu (MAE): {mae:.2f} TL")
    logger.info(f"Fiyat Doğruluğu (MAPE): %{mape:.2f}")
    logger.info(f"Toplam Getiri: %{total_return:.2f}")
    logger.info(f"Başlangıç: {initial_capital:.2f} TL, Son: {capital:.2f} TL")
    
    # Kazanan/kaybeden işlem analizi
    winning_trades = [t for t in trades if t['pct'] > 0]
    losing_trades = [t for t in trades if t['pct'] <= 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pct'] for t in losing_trades]) if losing_trades else 0
    
    logger.info(f"Kazanma Oranı: %{win_rate:.2f}")
    logger.info(f"Ort. Kazanç: %{avg_win:.2f}")
    logger.info(f"Ort. Kayıp: %{avg_loss:.2f}")
    
    return {
        'tf': tf,
        'direction_accuracy': direction_accuracy,
        'mae': mae,
        'mape': mape,
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'num_trades': len(trades)
    }


def main():
    logger.info("=" * 60)
    logger.info("GMSTR TÜM ZAMAN DİLİMLERİ BACKTEST")
    logger.info("=" * 60)
    
    model_path = "price_prediction_GMSTR_1d_updated.pkl"
    
    results = []
    for tf, bars in [('1h', 1), ('4h', 4), ('1d', 1), ('1w', 1)]:
        result = backtest_tf(tf, model_path, period='1y', bars_per_period=bars)
        if result:
            results.append(result)
    
    # Özet tablo
    logger.info("\n" + "=" * 60)
    logger.info("ÖZET TABLO")
    logger.info("=" * 60)
    logger.info(f"{'Zaman Dilimi':<12} {'Yön Doğruluğu':<15} {'MAPE':<10} {'Getiri':<10} {'Kazanma Oranı':<12}")
    logger.info("-" * 60)
    
    for r in results:
        logger.info(f"{r['tf']:<12} %{r['direction_accuracy']:.1f}{'':<10} %{r['mape']:.1f}{'':<5} %{r['total_return']:.1f}{'':<5} %{r['win_rate']:.1f}")
    
    # Sonuçları kaydet
    results_df = pd.DataFrame(results)
    results_df.to_csv("gmstr_backtest_results.csv", index=False)
    logger.info(f"\nSonuçlar kaydedildi: gmstr_backtest_results.csv")


if __name__ == "__main__":
    main()