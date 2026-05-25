"""
GMSTR, BTC, ETH Model Doğruluk Testi
1h, 4h, 1gün, 1hafta zaman dilimlerinde kapsamlı test
"""

import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime
from exchange_client_crypto import CryptoExchangeClient
from exchange_client_bist import BISTExchangeClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

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


def prepare_features_gmstr(df):
    """GMSTR için feature hesapla"""
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
    
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum']
    
    return df, features


def prepare_features_crypto(df):
    """BTC/ETH için feature hesapla"""
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
    df['stochastic'] = calculate_stochastic(df)
    df['williams_r'] = calculate_williams_r(df)
    
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                'stochastic', 'williams_r']
    
    return df, features


def prepare_features_eth(df):
    """ETH için özel feature hesapla (CCI ve OBV dahil)"""
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
    df['stochastic'] = calculate_stochastic(df)
    df['williams_r'] = calculate_williams_r(df)
    df['obv'] = calculate_obv(df)
    df['cci'] = calculate_cci(df)
    
    features = ['rsi', 'atr', 'z_score', 'volume_delta', 'macd', 'ema', 
                'bollinger_upper', 'bollinger_lower', 'sma', 'momentum',
                'stochastic', 'williams_r', 'obv', 'cci']
    
    return df, features


def test_model_accuracy(symbol, model_path, timeframe_hours, df, features, client_type="crypto"):
    """
    Model doğruluğunu test et
    
    Args:
        symbol: Sembol adı (BTCUSDT, ETHUSDT, GMSTR)
        model_path: Model dosya yolu
        timeframe_hours: Tahmin zaman dilimi (saat cinsinden)
        df: Hazırlanmış veri DataFrame'i
        features: Kullanılan feature'lar
        client_type: "crypto" veya "bist"
    """
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.warning(f"Model yüklenemedi: {model_path} - {e}")
        return None
    
    # Test verisi hazırla (son 200 bar)
    test_data = df.iloc[-200:].copy()
    
    if len(test_data) < 50:
        logger.warning(f"Test için yeterli veri yok: {len(test_data)} bar")
        return None
    
    predictions = []
    actuals = []
    signals = []
    entry_prices = []
    exit_prices = []
    
    for i in range(len(test_data) - timeframe_hours):
        X = test_data[features].iloc[i:i+1]
        current_price = test_data['close'].iloc[i]
        
        if pd.isna(X).any().any():
            continue
        
        try:
            pred_price = model.predict(X)[0]
            actual_price = test_data['close'].iloc[i + timeframe_hours]
            
            predictions.append(pred_price)
            actuals.append(actual_price)
            
            # Sinyal: tahmin > mevcut fiyat ise AL, değilse SAT
            signal = 1 if pred_price > current_price else -1
            signals.append(signal)
            entry_prices.append(current_price)
            exit_prices.append(actual_price)
            
        except Exception as e:
            continue
    
    if len(predictions) < 10:
        logger.warning(f"Yetersiz tahmin sayısı: {len(predictions)}")
        return None
    
    # Metrikleri hesapla
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    signals = np.array(signals)
    entry_prices = np.array(entry_prices)
    exit_prices = np.array(exit_prices)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    # MAPE hesapla (sıfıra bölme hatasından kaçın)
    mask = actuals != 0
    if mask.any():
        mape = mean_absolute_percentage_error(actuals[mask], predictions[mask]) * 100
    else:
        mape = float('inf')
    
    # Yön doğruluğu
    actual_direction = np.sign(actuals - entry_prices)
    predicted_direction = np.sign(predictions - entry_prices)
    direction_accuracy = np.sum(actual_direction == predicted_direction) / len(actual_direction) * 100
    
    # Kar/Zarar simülasyonu
    commission = 0.001  # %0.1 komisyon
    slippage = 0.0005   # %0.05 slippage
    position_size = 0.1  # Her işlemde 0.1 birim
    
    total_profit = 0
    winning_trades = 0
    losing_trades = 0
    
    for i in range(len(signals)):
        if signals[i] == 1:  # AL sinyali
            profit = (exit_prices[i] - entry_prices[i]) * position_size
        else:  # SAT sinyali
            profit = (entry_prices[i] - exit_prices[i]) * position_size
        
        # Komisyon ve slippage düş
        profit -= (entry_prices[i] * position_size * commission)
        profit -= (entry_prices[i] * position_size * slippage)
        
        total_profit += profit
        
        if profit > 0:
            winning_trades += 1
        elif profit < 0:
            losing_trades += 1
    
    total_trades = len(signals)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'symbol': symbol,
        'timeframe': f"{timeframe_hours}h",
        'model_path': model_path,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'profit_percentage': (total_profit / (entry_prices.sum() * position_size)) * 100 if entry_prices.sum() > 0 else 0,
        'avg_prediction': np.mean(predictions),
        'avg_actual': np.mean(actuals),
        'prediction_samples': len(predictions)
    }


def test_gmstr_models():
    """GMSTR modellerini test et"""
    logger.info("=" * 80)
    logger.info("GMSTR MODEL DOĞRULUK TESTLERİ")
    logger.info("=" * 80)
    
    client = BISTExchangeClient()
    
    # Veri çek (son 1000 bar - 1 saatlik)
    df = client.fetch_ohlcv("GMSTR", timeframe="1h", limit=1000)
    
    if df is None or df.empty:
        logger.error("GMSTR verisi çekilemedi")
        return []
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df, features = prepare_features_gmstr(df)
    df = df.dropna()
    
    results = []
    
    # Test edilecek modeller
    models_to_test = [
        ("1h", 1, "ai_trading_model_v4_GMSTR_1h.pkl"),
        ("4h", 4, "price_prediction_GMSTR_4h_updated.pkl"),
        ("4h (eski)", 4, "price_prediction_GMSTR_4h.pkl"),
        ("1gün", 24, "price_prediction_GMSTR_1d_updated.pkl"),
        ("1gün (eski)", 24, "price_prediction_GMSTR_1d.pkl"),
        ("1hafta", 168, "price_prediction_GMSTR_1d_updated.pkl"),  # 1hafta için 1d modelini kullan
    ]
    
    for timeframe_name, timeframe_hours, model_path in models_to_test:
        logger.info(f"\n--- GMSTR {timeframe_name} Model Testi ---")
        result = test_model_accuracy("GMSTR", model_path, timeframe_hours, df, features, "bist")
        if result:
            results.append(result)
            print_model_result(result)
        else:
            logger.warning(f"GMSTR {timeframe_name} modeli test edilemedi")
    
    return results


def test_btc_models():
    """BTC modellerini test et"""
    logger.info("=" * 80)
    logger.info("BTC MODEL DOĞRULUK TESTLERİ")
    logger.info("=" * 80)
    
    client = CryptoExchangeClient()
    
    # Veri çek (son 1000 bar - 1 saatlik)
    df = client.fetch_ohlcv("BTCUSDT", timeframe="1h", limit=1000)
    
    if df is None or df.empty:
        logger.error("BTC verisi çekilemedi")
        return []
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df, features = prepare_features_crypto(df)
    df = df.dropna()
    
    results = []
    
    # Test edilecek modeller
    models_to_test = [
        ("4h", 4, "price_prediction_BTC_4h_improved.pkl"),
        ("4h (eski)", 4, "price_prediction_BTC_4h.pkl"),
        ("4h (2years)", 4, "price_prediction_BTC_4h_2years.pkl"),
        ("1gün", 24, "price_prediction_BTC_1d_improved.pkl"),
        ("1gün (eski)", 24, "price_prediction_BTC_1d.pkl"),
        ("1gün (2years)", 24, "price_prediction_BTC_1d_2years.pkl"),
        ("1hafta", 168, "price_prediction_BTC_1d_improved.pkl"),  # 1hafta için 1d modelini kullan
    ]
    
    for timeframe_name, timeframe_hours, model_path in models_to_test:
        logger.info(f"\n--- BTC {timeframe_name} Model Testi ---")
        result = test_model_accuracy("BTCUSDT", model_path, timeframe_hours, df, features, "crypto")
        if result:
            results.append(result)
            print_model_result(result)
        else:
            logger.warning(f"BTC {timeframe_name} modeli test edilemedi")
    
    return results


def test_eth_models():
    """ETH modellerini test et"""
    logger.info("=" * 80)
    logger.info("ETH MODEL DOĞRULUK TESTLERİ")
    logger.info("=" * 80)
    
    client = CryptoExchangeClient()
    
    # Veri çek (son 1000 bar - 1 saatlik)
    df = client.fetch_ohlcv("ETHUSDT", timeframe="1h", limit=1000)
    
    if df is None or df.empty:
        logger.error("ETH verisi çekilemedi")
        return []
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df, features = prepare_features_eth(df)
    df = df.dropna()
    
    results = []
    
    # Test edilecek modeller
    models_to_test = [
        ("4h", 4, "price_prediction_ETH_4h.pkl"),
        ("1gün", 24, "price_prediction_ETH_1d.pkl"),
        ("1hafta", 168, "price_prediction_ETH_1d.pkl"),  # 1hafta için 1d modelini kullan
    ]
    
    for timeframe_name, timeframe_hours, model_path in models_to_test:
        logger.info(f"\n--- ETH {timeframe_name} Model Testi ---")
        result = test_model_accuracy("ETHUSDT", model_path, timeframe_hours, df, features, "crypto")
        if result:
            results.append(result)
            print_model_result(result)
        else:
            logger.warning(f"ETH {timeframe_name} modeli test edilemedi")
    
    return results


def print_model_result(result):
    """Model test sonucunu yazdır"""
    logger.info(f"  Model: {result['model_path']}")
    logger.info(f"  Tahmin Sayısı: {result['prediction_samples']}")
    logger.info(f"  MAE: {result['mae']:.4f}")
    logger.info(f"  RMSE: {result['rmse']:.4f}")
    logger.info(f"  MAPE: %{result['mape']:.2f}")
    logger.info(f"  Yön Doğruluğu: %{result['direction_accuracy']:.2f}")
    logger.info(f"  Toplam İşlem: {result['total_trades']}")
    logger.info(f"  Kazançlı İşlem: {result['winning_trades']}")
    logger.info(f"  Kayıplı İşlem: {result['losing_trades']}")
    logger.info(f"  Kazanma Oranı: %{result['win_rate']:.2f}")
    logger.info(f"  Net Kar/Zarar: ${result['total_profit']:.2f}")
    logger.info(f"  Kar Oranı: %{result['profit_percentage']:.2f}")
    logger.info(f"  Ortalama Tahmin: {result['avg_prediction']:.4f}")
    logger.info(f"  Ortalama Gerçek: {result['avg_actual']:.4f}")


def print_summary_report(all_results):
    """Özet rapor yazdır"""
    logger.info("=" * 80)
    logger.info("GENEL ÖZET RAPOR")
    logger.info("=" * 80)
    
    # Varlığa göre grupla
    symbols = {}
    for r in all_results:
        symbol = r['symbol']
        if symbol not in symbols:
            symbols[symbol] = []
        symbols[symbol].append(r)
    
    for symbol, results in symbols.items():
        logger.info(f"\n--- {symbol} Özet ---")
        
        # En iyi modeli bul (direction_accuracy'ye göre)
        best_model = max(results, key=lambda x: x['direction_accuracy'])
        logger.info(f"  En İyi Model (Yön Doğruluğu): {best_model['timeframe']} - %{best_model['direction_accuracy']:.2f}")
        logger.info(f"    Model: {best_model['model_path']}")
        
        # En karlı modeli bul
        most_profitable = max(results, key=lambda x: x['total_profit'])
        logger.info(f"  En Karlı Model: {most_profitable['timeframe']} - ${most_profitable['total_profit']:.2f}")
        logger.info(f"    Model: {most_profitable['model_path']}")
        
        # Ortalama doğruluk
        avg_accuracy = np.mean([r['direction_accuracy'] for r in results])
        logger.info(f"  Ortalama Yön Doğruluğu: %{avg_accuracy:.2f}")
        
        # Toplam kar/zarar
        total_profit = sum([r['total_profit'] for r in results])
        logger.info(f"  Toplam Kar/Zarar: ${total_profit:.2f}")


def main():
    """Ana test fonksiyonu"""
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "MODEL DOĞRULUK TEST BAŞLATILIYOR" + " " * 25 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info(f"Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    # GMSTR testleri
    gmstr_results = test_gmstr_models()
    all_results.extend(gmstr_results)
    
    # BTC testleri
    btc_results = test_btc_models()
    all_results.extend(btc_results)
    
    # ETH testleri
    eth_results = test_eth_models()
    all_results.extend(eth_results)
    
    # Özet rapor
    if all_results:
        print_summary_report(all_results)
        
        # Sonuçları dosyaya kaydet
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("model_accuracy_results.csv", index=False, encoding='utf-8-sig')
        logger.info("\nSonuçlar 'model_accuracy_results.csv' dosyasına kaydedildi.")
    else:
        logger.warning("Hiçbir model test edilemedi. Model dosyalarının mevcut olduğundan emin olun.")
    
    logger.info("\nTest tamamlandı!")


if __name__ == "__main__":
    main()