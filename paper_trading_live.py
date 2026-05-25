"""
Paper Trading Live Application
Gerçek zamanlı paper trading (sahte para)
1 hafta çalışacak şekilde ayarlanmış
"""

import pandas as pd
import numpy as np
import logging
import sys
import time
from datetime import datetime
from risk_management import RiskManager
from ai_trading_model_v4 import AITradingModelV4

from config import (
    COMMISSION_RATE,
    ATR_PERIOD,
    TRAILING_ATR_MULTIPLIER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class PaperTradingLive:
    """Paper Trading Live Application"""
    
    def __init__(self, symbols, timeframe="4h", risk_per_trade=0.005, initial_balance=500.0):
        self.symbols = symbols
        self.timeframe = timeframe
        self.risk_per_trade = risk_per_trade
        self.initial_balance = initial_balance
        self.balance_usdt = initial_balance
        self.balance_assets = {symbol: 0.0 for symbol in symbols}
        self.last_buy_prices = {symbol: None for symbol in symbols}
        self.trades = []
        self.equity_curve = []
        
        self.commission_rate = 0.001  # Binance standart: %0.1
        self.slippage_rate = 0.0005  # Binance standart: %0.05
        
        self.models = {}
        self.last_data = {}
        
        logger.info("=" * 70)
        logger.info("PAPER TRADING LIVE APPLICATION")
        logger.info("=" * 70)
        logger.info(f"Semboller: {', '.join(symbols)}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Risk per Trade: %{risk_per_trade*100:.1f}")
        logger.info(f"Başlangıç Bakiyesi: ${initial_balance:.2f} USDT")
        logger.info("=" * 70)
        logger.info("")
    
    def load_models(self):
        """Modelleri yükle"""
        logger.info("Modeller yükleniyor...")
        
        for symbol in self.symbols:
            model_path = f"ai_trading_model_v4_{symbol}.pkl"
            ai_model = AITradingModelV4(model_path=model_path)
            
            if ai_model.load_model():
                self.models[symbol] = ai_model
                logger.info(f"Model yüklendi: {symbol}")
            else:
                logger.error(f"Model bulunamadı: {symbol}")
                return False
        
        logger.info("Tüm modeller yüklendi!")
        logger.info("")
        return True
    
    def fetch_data(self):
        """Veri çek"""
        from exchange_client_stocks import StockCommodityExchangeClient
        from exchange_client_bist import BISTExchangeClient
        
        us_client = StockCommodityExchangeClient()
        bist_client = BISTExchangeClient()
        
        for symbol in self.symbols:
            # BIST sembolleri için BIST client kullan
            if symbol in ["THYAO", "TUPRS", "KCHOL", "SISE", "AKBNK", "GARAN", "ISCTR", "BIMAS", "EREGL", "SAHOL", "TCELL", "FROTO", "TOASO", "KRDMD", "YKBNK"]:
                df = bist_client.fetch_ohlcv(symbol, self.timeframe, limit=200)
            else:
                df = us_client.fetch_ohlcv(symbol, self.timeframe, limit=200)
            
            if df is not None and not df.empty:
                df['atr'] = df['high'] - df['low']
                df['atr'] = df['atr'].rolling(window=ATR_PERIOD).mean()
                self.last_data[symbol] = df
            else:
                logger.error(f"Veri çekilemedi: {symbol}")
        
        return True
    
    def process_signals(self):
        """Sinyalleri işle"""
        for symbol in self.symbols:
            if symbol not in self.last_data or symbol not in self.models:
                continue
            
            df = self.last_data[symbol]
            model = self.models[symbol]
            
            if len(df) < 50:
                continue
            
            window = df.iloc[-50:].copy()
            
            price = window['close'].iloc[-1]
            high = window['high'].iloc[-1]
            low = window['low'].iloc[-1]
            atr = window['atr'].iloc[-1]
            
            # Tarih bilgisi
            timestamp = window['timestamp'].iloc[-1]
            date_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M')
            
            # AI modeli ile tahmin yap
            signal = model.predict(window, probability_threshold=0.50)
            
            # Stop-loss kontrolü
            if self.balance_assets[symbol] > 0 and self.last_buy_prices[symbol]:
                stop_loss_price = self.last_buy_prices[symbol] - (1.0 * atr)
                
                if low <= stop_loss_price:
                    sell_price = price * (1 - self.slippage_rate)
                    revenue = self.balance_assets[symbol] * sell_price
                    commission = revenue * self.commission_rate
                    net_revenue = revenue - commission
                    profit = net_revenue - (self.balance_assets[symbol] * self.last_buy_prices[symbol])
                    self.balance_usdt += net_revenue
                    
                    self.trades.append({
                        'type': 'SELL (STOP)',
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'date': date_str,
                        'price': sell_price,
                        'amount': self.balance_assets[symbol],
                        'profit': profit,
                        'balance_usdt': self.balance_usdt,
                    })
                    
                    logger.info(f"📉 SELL (STOP) | {symbol} | {date_str}")
                    logger.info(f"   Fiyat: ${sell_price:.4f}")
                    logger.info(f"   Miktar: {self.balance_assets[symbol]:.6f} {symbol}")
                    logger.info(f"   Kar/Zarar: ${profit:.4f}")
                    logger.info(f"   Kasa: ${self.balance_usdt:.2f} USDT")
                    logger.info("")
                    
                    self.balance_assets[symbol] = 0.0
                    self.last_buy_prices[symbol] = None
                    continue
            
            # Buy sinyali
            if signal == "BUY" and self.balance_assets[symbol] == 0:
                risk_capital = self.balance_usdt * self.risk_per_trade
                stop_distance = 1.0 * atr
                
                # Position size hesapla (Risk management)
                position_size = risk_capital / stop_distance
                
                # Toplam maliyet kontrolü
                buy_price = price * (1 + self.slippage_rate)
                cost = position_size * buy_price
                commission = cost * self.commission_rate
                total_cost = cost + commission
                
                # Kasa bakiyesini aşmamasını sağla
                if total_cost > self.balance_usdt:
                    # Kasa bakiyesini aşarsa, position size'ı küçült
                    position_size = (self.balance_usdt * 0.95) / buy_price
                    cost = position_size * buy_price
                    commission = cost * self.commission_rate
                    total_cost = cost + commission
                
                self.balance_usdt -= total_cost
                self.balance_assets[symbol] += position_size
                self.last_buy_prices[symbol] = buy_price
                
                self.trades.append({
                    'type': 'BUY',
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'date': date_str,
                    'price': buy_price,
                    'amount': position_size,
                    'risk_capital': risk_capital,
                    'profit': -commission,
                    'balance_usdt': self.balance_usdt,
                })
                
                logger.info(f"📈 BUY | {symbol} | {date_str}")
                logger.info(f"   Fiyat: ${buy_price:.4f}")
                logger.info(f"   Miktar: {position_size:.6f} {symbol}")
                logger.info(f"   Risk: ${risk_capital:.2f}")
                logger.info(f"   Kasa: ${self.balance_usdt:.2f} USDT")
                logger.info("")
            
            # Sell sinyali
            elif signal == "SELL" and self.balance_assets[symbol] > 0:
                sell_price = price * (1 - self.slippage_rate)
                revenue = self.balance_assets[symbol] * sell_price
                commission = revenue * self.commission_rate
                net_revenue = revenue - commission
                profit = net_revenue - (self.balance_assets[symbol] * self.last_buy_prices[symbol])
                self.balance_usdt += net_revenue
                
                self.trades.append({
                    'type': 'SELL',
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'date': date_str,
                    'price': sell_price,
                    'amount': self.balance_assets[symbol],
                    'profit': profit,
                    'balance_usdt': self.balance_usdt,
                })
                
                logger.info(f"📉 SELL | {symbol} | {date_str}")
                logger.info(f"   Fiyat: ${sell_price:.4f}")
                logger.info(f"   Miktar: {self.balance_assets[symbol]:.6f} {symbol}")
                logger.info(f"   Kar/Zarar: ${profit:.4f}")
                logger.info(f"   Kasa: ${self.balance_usdt:.2f} USDT")
                logger.info("")
                
                self.balance_assets[symbol] = 0.0
                self.last_buy_prices[symbol] = None
    
    def calculate_equity(self):
        """Toplam varlığı hesapla"""
        total_equity = self.balance_usdt
        
        for symbol in self.symbols:
            if symbol in self.last_data and self.balance_assets[symbol] > 0:
                price = self.last_data[symbol]['close'].iloc[-1]
                total_equity += self.balance_assets[symbol] * price
        
        self.equity_curve.append(total_equity)
        return total_equity
    
    def print_summary(self):
        """Özet yazdır"""
        total_equity = self.calculate_equity()
        total_return = total_equity - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        logger.info("=" * 70)
        logger.info("GÜNCEL DURUM")
        logger.info("=" * 70)
        logger.info(f"Toplam Varlık: ${total_equity:.2f} USDT")
        logger.info(f"Net Getiri: ${total_return:.2f} USDT (%{return_pct:.2f})")
        logger.info(f"Kasa (USDT): ${self.balance_usdt:.2f}")
        logger.info(f"Toplam İşlem: {len(self.trades)}")
        logger.info("")
        
        # Sembol bazlı durum
        logger.info("Sembol Bazlı Durum:")
        for symbol in self.symbols:
            if self.balance_assets[symbol] > 0:
                if symbol in self.last_data:
                    price = self.last_data[symbol]['close'].iloc[-1]
                    asset_value = self.balance_assets[symbol] * price
                    logger.info(f"  {symbol}: {self.balance_assets[symbol]:.6f} × ${price:.4f} = ${asset_value:.2f}")
            else:
                logger.info(f"  {symbol}: 0.000000")
        
        logger.info("=" * 70)
        logger.info("")
    
    def run(self, duration_hours=168):  # 1 hafta = 168 saat
        """Paper trading çalıştır"""
        logger.info(f"Paper Trading Başlatılıyor... ({duration_hours} saat)")
        logger.info("")
        
        if not self.load_models():
            logger.error("Modeller yüklenemedi")
            return
        
        start_time = time.time()
        iteration = 0
        
        while True:
            iteration += 1
            current_time = time.time()
            elapsed_hours = (current_time - start_time) / 3600
            
            logger.info(f"--- İterasyon {iteration} | Geçen Süre: {elapsed_hours:.1f} saat ---")
            logger.info("")
            
            # Veri çek
            self.fetch_data()
            
            # Sinyalleri işle
            self.process_signals()
            
            # Özet yazdır
            self.print_summary()
            
            # Süre kontrolü
            if elapsed_hours >= duration_hours:
                logger.info("Süre doldu! Paper Trading sonlandırılıyor...")
                break
            
            # Bekle (15m timeframe için 15 dakika = 900 saniye)
            logger.info("Sonraki kontrol: 15 dakika sonra...")
            logger.info("")
            time.sleep(900)  # 15 dakika


if __name__ == "__main__":
    # 15 ABD + 15 Türk hisse senedi ile paper trading (7/24)
    us_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ", "AMD", "INTC", "CSCO", "CRM", "ORCL"]
    tr_symbols = ["THYAO", "TUPRS", "KCHOL", "SISE", "AKBNK", "GARAN", "ISCTR", "BIMAS", "EREGL", "SAHOL", "TCELL", "FROTO", "TOASO", "KRDMD", "YKBNK"]
    symbols = us_symbols + tr_symbols
    
    paper_trading = PaperTradingLive(
        symbols=symbols,
        timeframe="15m",
        risk_per_trade=0.005,
        initial_balance=500.0
    )
    
    # 1 hafta çalıştır
    paper_trading.run(duration_hours=168)
