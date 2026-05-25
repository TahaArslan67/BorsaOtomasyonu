import logging
from datetime import datetime, date
from config import (
    INITIAL_BALANCE_USDT,
    POSITION_SIZE_PERCENT,
    MAX_DAILY_LOSS_PERCENT,
    COOLDOWN_BARS,
    TIME_EXIT_BARS,
    TRAILING_ATR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class PaperPortfolio:
    def __init__(self):
        self.balance_usdt = INITIAL_BALANCE_USDT
        self.balance_asset = 0.0
        self.initial_balance = INITIAL_BALANCE_USDT
        self.position_size_percent = POSITION_SIZE_PERCENT
        self.last_buy_price = None

        self.daily_start_balance = INITIAL_BALANCE_USDT
        self.current_date = date.today()
        self.max_daily_loss_percent = MAX_DAILY_LOSS_PERCENT

        self.last_atr = None
        self.trailing_atr_multiplier = TRAILING_ATR_MULTIPLIER

        # Short pozisyon takibi
        self.short_position = False
        self.short_entry_price = None
        self.short_size = 0.0

        # Cool-down: Stop sonrasi bekleme
        self.cooldown_counter = 0
        self.cooldown_bars = COOLDOWN_BARS

        # Time-Exit: Islem maksimum bar suresi (V4.0: 36 bar = 3 saat)
        self.entry_bar_index = None
        self.time_exit_bars = TIME_EXIT_BARS

        # Trend Rider: 2.5 * ATR trailing stop-loss (Chandelier Exit)
        self.stop_price = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        # Piramit (Scaling In) takibi
        self.scaling_in_enabled = True
        self.scaling_in_profit_threshold = 0.02  # %2 kâr
        self.scaling_in_size_ratio = 0.5  # Mevcut pozisyonun %50'si kadar ekle
        self.last_scaling_in_bar = None
        self.original_position_size = 0.0
        self.original_entry_price = 0.0

    def get_position_value(self, price):
        return self.balance_asset * price

    def get_total_value(self, price):
        return self.balance_usdt + self.get_position_value(price)

    def calculate_position_size(self, price):
        risk_capital = self.get_total_value(price) * (self.position_size_percent / 100)
        return risk_capital / price

    def _update_daily_balance(self, price):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.daily_start_balance = self.get_total_value(price)
            logger.info(f"Yeni gun basladi. Gunluk baslangic bakiyesi: {self.daily_start_balance:.2f} USDT")

    def can_trade(self, price):
        self._update_daily_balance(price)

        # Cool-down kontrolu
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                logger.info("Cool-down bitti, yeni islem acilabilir.")
            return False

        current_value = self.get_total_value(price)
        loss_pct = (self.daily_start_balance - current_value) / self.daily_start_balance * 100
        if loss_pct >= self.max_daily_loss_percent:
            logger.warning(
                f"Gunluk max zarar limiti asildi! Baslangic: {self.daily_start_balance:.2f} | "
                f"Anlik: {current_value:.2f} | Zarar: %{loss_pct:.2f}. Bugun islem yapilmayacak."
            )
            return False
        return True

    def activate_cooldown(self):
        """Stop loss sonrasi cool-down baslat"""
        self.cooldown_counter = self.cooldown_bars
        logger.info(f"Cool-down aktif: {self.cooldown_bars} bar ({self.cooldown_bars * 5}dk) bekleniyor.")

    def buy(self, price, atr=None, bar_index=None):
        if self.balance_usdt <= 0:
            logger.warning("Yetersiz USDT bakiyesi.")
            return False

        if not self.can_trade(price):
            return False

        amount = self.calculate_position_size(price)
        cost = amount * price

        if cost > self.balance_usdt:
            cost = self.balance_usdt
            amount = cost / price

        self.balance_usdt -= cost
        self.balance_asset += amount
        self.last_buy_price = price
        self.last_atr = atr
        self.entry_bar_index = bar_index

        # Piramit için orijinal pozisyon bilgileri
        if self.balance_asset == amount:  # İlk giriş
            self.original_position_size = amount
            self.original_entry_price = price
            self.last_scaling_in_bar = None

        # Trend Rider: Chandelier Exit - 2.5 * ATR stop-loss
        self.stop_price = price - (atr * self.trailing_atr_multiplier) if atr else price * 0.975
        self.highest_price_since_entry = price
        self.lowest_price_since_entry = None

        # Cool-down reset
        self.cooldown_counter = 0

        sl_msg = f" | SL: {self.stop_price:.2f}" if self.stop_price else ""
        logger.info(
            f"[PAPER BUY] Fiyat: {price:.2f} | Miktar: {amount:.6f} | Maliyet: {cost:.2f} USDT | "
            f"Kalan USDT: {self.balance_usdt:.2f}{sl_msg}"
        )
        return True

    def check_scaling_in(self, price, atr, bar_index, st_direction):
        """Piramit: %2 kârda ve SuperTrend AL sinyali gelince %50 ekle"""
        if not self.scaling_in_enabled:
            return False
        
        if self.balance_asset <= 0 or self.last_scaling_in_bar == bar_index:
            return False
        
        # %2 kâr kontrolü
        if self.original_entry_price > 0:
            profit_pct = (price - self.original_entry_price) / self.original_entry_price
            if profit_pct >= self.scaling_in_profit_threshold:
                # SuperTrend AL sinyali kontrolü
                if st_direction == 1:  # Yukarı trend
                    # Mevcut pozisyonun %50'si kadar ekle
                    additional_amount = self.original_position_size * self.scaling_in_size_ratio
                    additional_cost = additional_amount * price
                    
                    if additional_cost <= self.balance_usdt:
                        self.balance_usdt -= additional_cost
                        self.balance_asset += additional_amount
                        self.last_scaling_in_bar = bar_index
                        
                        # Ortalama giriş fiyatını güncelle
                        total_cost = (self.original_position_size * self.original_entry_price) + additional_cost
                        total_amount = self.original_position_size + additional_amount
                        self.last_buy_price = total_cost / total_amount
                        
                        logger.info(
                            f"[PIRAMIT] %2 kârda + ST AL sinyali! Eklenen: {additional_amount:.6f} | "
                            f"Yeni ortalama fiyat: {self.last_buy_price:.2f} | Toplam miktar: {self.balance_asset:.6f}"
                        )
                        return True
        return False

    def sell(self, price, is_short_close=False):
        if self.balance_asset <= 0 and not self.short_position:
            logger.warning("Satilacak varlik yok.")
            return False

        if self.short_position and is_short_close:
            # Short pozisyon kapat
            profit = self.short_size * (self.short_entry_price - price)
            self.balance_usdt += profit
            logger.info(
                f"[PAPER CLOSE SHORT] Fiyat: {price:.2f} | Entry: {self.short_entry_price:.2f} | "
                f"Kar/Zarar: {profit:.2f} USDT | Toplam USDT: {self.balance_usdt:.2f}"
            )
            self.short_position = False
            self.short_entry_price = None
            self.short_size = 0.0
            self.stop_price = None
            self.highest_price_since_entry = None
            self.lowest_price_since_entry = None
            return True

        revenue = self.balance_asset * price
        profit = 0.0
        if self.last_buy_price:
            profit = revenue - (self.balance_asset * self.last_buy_price)

        self.balance_usdt += revenue
        sold_amount = self.balance_asset
        self.balance_asset = 0.0
        self.last_buy_price = None
        self.stop_price = None
        self.highest_price_since_entry = None
        self.lowest_price_since_entry = None
        self.short_position = False
        
        # Piramit değişkenlerini sıfırla
        self.original_position_size = 0.0
        self.original_entry_price = 0.0
        self.last_scaling_in_bar = None

        logger.info(
            f"[PAPER SELL] Fiyat: {price:.2f} | Miktar: {sold_amount:.6f} | Gelir: {revenue:.2f} USDT | "
            f"Kar/Zarar: {profit:.2f} USDT | Toplam USDT: {self.balance_usdt:.2f}"
        )
        return True

    def open_short(self, price, atr=None, bar_index=None):
        """Short pozisyon ac (simulasyon)"""
        if not self.can_trade(price):
            return False

        risk_capital = self.get_total_value(price) * (self.position_size_percent / 100)
        self.short_size = risk_capital / price
        self.short_entry_price = price
        self.short_position = True
        self.last_atr = atr
        self.entry_bar_index = bar_index

        # Trend Rider: Chandelier Exit - 3.0 * ATR stop-loss
        self.stop_price = price + (atr * self.trailing_atr_multiplier) if atr else price * 1.03
        self.lowest_price_since_entry = price
        self.highest_price_since_entry = None

        # Cool-down reset
        self.cooldown_counter = 0

        sl_msg = f" | SL: {self.stop_price:.2f}" if self.stop_price else ""
        logger.info(
            f"[PAPER OPEN SHORT] Fiyat: {price:.2f} | Miktar: {self.short_size:.6f}{sl_msg}"
        )
        return True

    def check_stop_loss(self, price):
        # Long pozisyon Chandelier Exit stop kontrolu
        if self.balance_asset > 0 and self.stop_price:
            if price <= self.stop_price:
                loss_pct = (self.last_buy_price - price) / self.last_buy_price * 100 if self.last_buy_price else 0
                logger.warning(
                    f"Chandelier Stop! Alis: {self.last_buy_price:.2f} | "
                    f"Stop: {self.stop_price:.2f} | Anlik: {price:.2f} | Zarar: %{loss_pct:.2f}"
                )
                self.activate_cooldown()
                return True

        # Short pozisyon Chandelier Exit stop kontrolu
        if self.short_position and self.stop_price:
            if price >= self.stop_price:
                loss_pct = (price - self.short_entry_price) / self.short_entry_price * 100 if self.short_entry_price else 0
                logger.warning(
                    f"Short Chandelier Stop! Entry: {self.short_entry_price:.2f} | "
                    f"Stop: {self.stop_price:.2f} | Anlik: {price:.2f} | Zarar: %{loss_pct:.2f}"
                )
                self.activate_cooldown()
                return True
        return False

    def update_trailing_stop(self, price, atr=None):
        """Trend Rider: Chandelier Exit - 2.5 * ATR trailing stop guncelle"""
        atr_val = atr if atr is not None else self.last_atr
        if atr_val is None or atr_val <= 0:
            return

        mult = self.trailing_atr_multiplier

        # Long: En yuksek fiyati takip et, stop = highest - 2.5*ATR
        if self.balance_asset > 0 and self.last_buy_price:
            if self.highest_price_since_entry is None or price > self.highest_price_since_entry:
                self.highest_price_since_entry = price
            new_stop = self.highest_price_since_entry - (atr_val * mult)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                logger.debug(
                    f"Long trailing stop yukseltildi: {self.stop_price:.2f} "
                    f"(highest: {self.highest_price_since_entry:.2f})"
                )

        # Short: En dusuk fiyati takip et, stop = lowest + 2.5*ATR
        if self.short_position and self.short_entry_price:
            if self.lowest_price_since_entry is None or price < self.lowest_price_since_entry:
                self.lowest_price_since_entry = price
            new_stop = self.lowest_price_since_entry + (atr_val * mult)
            if new_stop < self.stop_price:
                self.stop_price = new_stop
                logger.debug(
                    f"Short trailing stop dusuruldu: {self.stop_price:.2f} "
                    f"(lowest: {self.lowest_price_since_entry:.2f})"
                )

    def check_time_exit(self, current_bar_index):
        """Time-Exit: Islem 72 bar (6 saat) icinde hedefine ulasmadiysa kapat"""
        if self.entry_bar_index is None or current_bar_index is None:
            return False

        bars_held = current_bar_index - self.entry_bar_index
        if bars_held >= self.time_exit_bars:
            if self.balance_asset > 0:
                logger.info(f"Time-Exit: Long pozisyon {bars_held} bar sonra kapatildi.")
            elif self.short_position:
                logger.info(f"Time-Exit: Short pozisyon {bars_held} bar sonra kapatildi.")
            return True
        return False

    def check_take_profit(self, price):
        """Trend Rider: Sabit TP yok. Tüm çıkış trailing stop üzerinden."""
        return False

    def has_position(self):
        """Pozisyon var mi? (Long veya Short)"""
        return self.balance_asset > 0 or self.short_position
