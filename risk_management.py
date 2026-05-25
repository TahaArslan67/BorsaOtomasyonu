"""
Risk Yönetimi Modülü
Dynamic Leverage, Kelly Criterion ve Circuit Breaker
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

class RiskManager:
    """
    Profesyonel Risk Yönetimi Sınıfı
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.max_leverage = 100.0
        self.min_leverage = 5.0
        self.max_drawdown_threshold = 0.15  # %15
        self.circuit_breaker_threshold = 0.05  # %5
        
    def calculate_dynamic_leverage(
        self, 
        atr: float, 
        price: float, 
        volatility_window: int = 14
    ) -> float:
        """
        Volatiliteye göre dinamik kaldıraç hesapla
        
        Formül:
        Leverage = min(max_leverage, (0.5 / (ATR/Price)) * safety_factor)
        """
        if atr is None or price <= 0:
            return self.min_leverage
        
        # Volatilite oranı (ATR/Price)
        volatility_ratio = atr / price
        
        # Güvenlik faktörü (volatilite arttıkça kaldıraç düşer)
        safety_factor = 0.5 / (1 + volatility_ratio * 10)
        
        # Dinamik kaldıraç
        dynamic_leverage = min(
            self.max_leverage,
            (safety_factor / volatility_ratio)
        )
        
        # Minimum kaldıraç garantisi
        return max(self.min_leverage, dynamic_leverage)
    
    def kelly_criterion_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_balance: float
    ) -> float:
        """
        Kelly Criterion ile pozisyon büyüklüğü hesapla
        
        Kelly % = (p * b - q) / b
        p = win rate
        q = loss rate = 1 - p
        b = avg_win / avg_loss (odds)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Odds (kazanç/kayıp oranı)
        odds = avg_win / abs(avg_loss)
        
        # Kelly yüzdesi
        kelly_percent = (win_rate * odds - (1 - win_rate)) / odds
        
        # Kelly'nin %50'sini kullan (fractional Kelly)
        fractional_kelly = max(0, kelly_percent * 0.5)
        
        # Pozisyon büyüklüğü
        position_size = current_balance * fractional_kelly
        
        return position_size
    
    def circuit_breaker_check(
        self,
        current_drawdown: float,
        consecutive_losses: int,
        max_consecutive_losses: int = 3
    ) -> Tuple[bool, str]:
        """
        Devre kesici kontrolü
        """
        # Drawdown kontrolü
        if current_drawdown > self.max_drawdown_threshold:
            return True, f"Drawdown limiti aşıldı: %{current_drawdown*100:.2f}"
        
        # Ardışık kayıp kontrolü
        if consecutive_losses >= max_consecutive_losses:
            return True, f"Ardışık kayıp limiti: {consecutive_losses}"
        
        return False, "OK"
    
    def calculate_tick_based_stop_loss(
        self,
        entry_price: float,
        atr: float,
        leverage: float,
        tick_size: float = 0.01
    ) -> Tuple[float, float]:
        """
        Tick bazlı stop-loss hesapla (mum içi likidasyon önleme)
        
        Returns:
            (stop_loss_price, liquidation_distance_percent)
        """
        if atr is None or entry_price <= 0:
            return entry_price * 0.95, 0.05
        
        # Stop-loss mesafesi (ATR * multiplier)
        stop_distance = atr * 2.0  # 2x ATR
        
        # Stop-loss fiyatı
        stop_loss_price = entry_price - stop_distance
        
        # Likidasyon mesafesi (kaldıraç bazlı)
        liquidation_distance = 1.0 / leverage
        
        # Stop-loss, likidasyondan uzak olmalı
        if stop_distance < liquidation_distance * 0.5:
            # Stop-loss'u likidasyondan uzaklaştır
            stop_loss_price = entry_price - (liquidation_distance * 0.5)
        
        # Tick size'a yuvarla
        stop_loss_price = round(stop_loss_price / tick_size) * tick_size
        
        # Mesafe yüzdesi
        distance_percent = (entry_price - stop_loss_price) / entry_price
        
        return stop_loss_price, distance_percent
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Sharpe Ratio hesapla
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Günlük risk-free rate
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Sortino Ratio hesapla (negatif volatilite odaklı)
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        # Sadece negatif getiriler
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_std * np.sqrt(252)
        return sortino
    
    def monte_carlo_simulation(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        num_trades: int,
        num_simulations: int = 10000
    ) -> dict:
        """
        Monte Carlo Simülasyonu
        """
        results = []
        
        for _ in range(num_simulations):
            # Rastgele işlemler
            trades = np.random.random(num_trades)
            wins = trades < win_rate
            losses = ~wins
            
            # Getiriler
            trade_returns = np.zeros(num_trades)
            trade_returns[wins] = avg_win
            trade_returns[losses] = avg_loss
            
            # Kümülatif getiri
            cumulative_return = np.sum(trade_returns)
            results.append(cumulative_return)
        
        results = np.array(results)
        
        return {
            'mean': np.mean(results),
            'std': np.std(results),
            'percentile_5': np.percentile(results, 5),
            'percentile_95': np.percentile(results, 95),
            'probability_positive': np.mean(results > 0)
        }


# Örnek kullanım
if __name__ == "__main__":
    rm = RiskManager()
    
    # Örnek: MATIC için dinamik kaldıraç
    atr = 0.05  # $0.05 ATR
    price = 1.0  # $1.00 MATIC
    leverage = rm.calculate_dynamic_leverage(atr, price)
    print(f"Dinamik Kaldıraç: {leverage:.2f}x")
    
    # Örnek: Kelly Criterion
    win_rate = 0.40
    avg_win = 253.0
    avg_loss = -167.0
    position_size = rm.kelly_criterion_position_size(win_rate, avg_win, avg_loss, 10000)
    print(f"Kelly Pozisyon Büyüklüğü: ${position_size:.2f}")
    
    # Örnek: Monte Carlo
    mc_results = rm.monte_carlo_simulation(win_rate, avg_win, avg_loss, 10)
    print(f"Monte Carlo - Pozitif Olasılık: {mc_results['probability_positive']:.2%}")
